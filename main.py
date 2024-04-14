import os
import random
import re
import string
import time
from collections import Counter
from collections.abc import Callable

import click
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset, ClassLabel
from dotenv import load_dotenv
from evaluate import load
from torchinfo import summary
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EarlyStoppingCallback,
    PreTrainedModel, PreTrainedTokenizer, Trainer,
    TrainingArguments, AutoConfig
)
from torch.utils.data import Dataset as TorchDataset
from transformers.integrations import NeptuneCallback

from fast_aug.flow import SequentialAugmenter, SelectorAugmenter, ChanceAugmenter
from fast_aug.text import (
    CharsRandomDeleteAugmenter, CharsRandomSwapAugmenter, CharsRandomInsertAugmenter, CharsRandomSubstituteAugmenter,
    WordsRandomDeleteAugmenter, WordsRandomSwapAugmenter, WordsRandomInsertAugmenter, WordsRandomSubstituteAugmenter
)


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE
print(f"IS_CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")


load_dotenv()
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


TASK_TO_FIELDS = {
    # glue
    "glue/cola": ("sentence", None),
    "glue/mnli": ("premise", "hypothesis"),
    "glue/mrpc": ("sentence1", "sentence2"),
    "glue/qnli": ("question", "sentence"),
    "glue/qqp": ("question1", "question2"),
    "glue/rte": ("sentence1", "sentence2"),
    "glue/sst2": ("sentence", None),
    "glue/stsb": ("sentence1", "sentence2"),
    "glue/wnli": ("sentence1", "sentence2"),
    # super_glue
    "super_glue/boolq": ("question", "passage"),
    "super_glue/cb": ("premise", "hypothesis"),
    "super_glue/rte": ("premise", "hypothesis"),
    "super_glue/wic": ("sentence1", "sentence2"),
    "super_glue/wsc": ("text", None),
    "super_glue/axb": ("premise", "hypothesis"),
    "super_glue/axg": ("premise", "hypothesis"),
    # local
    "senti_comments": ("text", None),
    "serbmr_3c": ("text", None),
    "sts_news": ("text_1", "text_2"),
}
GLUE_TASK_TO_MAIN_METRIC = {
    # glue
    "glue/cola": "matthews_correlation",
    "glue/mnli": "accuracy",
    "glue/mrpc": "accuracy",
    "glue/qnli": "accuracy",
    "glue/qqp": "accuracy",
    "glue/rte": "accuracy",
    "glue/sst2": "accuracy",
    "glue/stsb": "pearson",
    "glue/wnli": "accuracy",
    # super glue
    "super_glue/boolq": "accuracy",
    "super_glue/cb": "f1",
    "super_glue/rte": "accuracy",
    "super_glue/wic": "accuracy",
    "super_glue/wsc": "accuracy",
    "super_glue/axb": "matthews_correlation",
    "super_glue/axg": "accuracy",
    # local
    "senti_comments": "f1",
    "serbmr_3c": "f1",
    "sts_news": "f1",
}
GLUE_TASK_TO_NUM_LABELS = {
    # glue
    "glue/cola": 2,
    "glue/mnli": 3,
    "glue/mrpc": 2,
    "glue/qnli": 2,
    "glue/qqp": 2,
    "glue/rte": 2,
    "glue/sst2": 2,
    "glue/stsb": 1,
    "glue/wnli": 2,
    # super glue
    "super_glue/boolq": 2,
    "super_glue/cb": 3,
    "super_glue/rte": 2,
    "super_glue/wic": 2,
    "super_glue/wsc": 2,
    "super_glue/axb": 2,
    "super_glue/axg": 2,
    # local
    "senti_comments": 6,
    "serbmr_3c": 3,
    "sts_news": 6,
}


class AugmentedTokenizedDataset(TorchDataset):
    def __init__(
            self,
            dataset: Dataset,
            tokenizer: PreTrainedTokenizer,
            text_field_1: str,
            text_field_2: str | None = None,
            augmentation_pipeline: Callable | None = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_field_1 = text_field_1
        self.text_field_2 = text_field_2
        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]

        text_1 = example[self.text_field_1]
        text_2 = example[self.text_field_2] if self.text_field_2 else None

        if self.augmentation_pipeline:
            text_1 = self.augmentation_pipeline(text_1)
            text_2 = self.augmentation_pipeline(text_2) if text_2 else None

        tokenized_example = self.tokenizer(
            text_1,
            text_2,
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        return {
            **tokenized_example,
            "label": example["label"],
        }


def get_low_resource_dataset_texts() -> list[str]:
    low_resource_dataset = load_dataset("tatoeba", "en-mr", split="train")
    return list({row['en'] for row in low_resource_dataset['translation']})  # Removing duplicates list->set->list


def sample_dataset_as_low_resource(
        dataset: Dataset,
        column_name: str,
        *,
        sample_size: int | None = None,
        sample_share: float | None = None,
) -> Dataset:
    assert (sample_size is None) != (sample_share is None), "either sample_size or sample_share should be provided"
    sample_size = sample_size or int(len(dataset) * sample_share)
    assert 0 < sample_size <= len(dataset), "sample_size should be in (0, len(dataset)]"

    # get low resource dataset texts distribution
    low_resource_texts = get_low_resource_dataset_texts()
    _length_counts = Counter([len(text.split()) for text in low_resource_texts])
    _total_sentences = sum(_length_counts.values())
    low_resource_texts_distribution = {length: count / _total_sentences for length, count in _length_counts.items()}

    # select indexes to sample
    df = pd.DataFrame({
        'idx': range(len(dataset)),
        'length': [len(text.split()) for text in dataset[column_name]],
        'sampling_weights': [low_resource_texts_distribution.get(len(text.split()), 1e-5) for text in dataset[column_name]],
    })
    sampled_indexes = df.sample(n=sample_size, weights='sampling_weights', replace=False, random_state=SEED)['idx']

    return dataset.select(sampled_indexes)


def load_glue_dataset(task_name: str) -> DatasetDict:
    dataset = load_dataset(*task_name.split("/"))

    if task_name == "glue/mnli":
        # rename splits for MNLI
        dataset["validation"] = dataset["validation_matched"]
        dataset["test"] = dataset["test_matched"]
        del dataset["validation_matched"], dataset["test_matched"]

    return dataset


def load_local_senti_comments_dataset() -> DatasetDict:
    dataset = load_dataset(
        'csv',
        data_files='datasets/SentiComments.SR.corr.txt',
        delimiter='\t',
        column_names=['label', 'id', 'text'],
        verification_mode='all_checks',
        split='train',
    )
    # remove letter s (sarcasm mark) from labels
    dataset = dataset.map(lambda example: {'label': example['label'].replace('s', '')})
    assert len(dataset.unique('label')) == 6, "Expected 6 unique labels"
    # convert labels to ClassLabel
    class_label = ClassLabel(num_classes=6, names=['+1', '-1', '+M', '-M', '+NS', '-NS'])
    dataset = dataset.cast_column('label', class_label)
    # split into train and validation
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
    dataset = DatasetDict({'train': dataset['train'], 'validation': dataset['test']})
    return dataset


def load_local_serbmr_3c_dataset() -> DatasetDict:
    dataset = load_dataset(
        'csv',
        data_files='datasets/SerbMR-3C.csv',
        column_names=['Text', 'class-att'],
        verification_mode='all_checks',
        split='train',
        skiprows=1,
    )
    dataset = dataset.rename_columns({'Text': 'text', 'class-att': 'label'})
    assert len(dataset.unique('label')) == 3, "Expected 3 unique labels"
    # convert labels to ClassLabel
    class_label = ClassLabel(num_classes=3, names=['POSITIVE', 'NEUTRAL', 'NEGATIVE'])
    dataset = dataset.cast_column('label', class_label)
    # split into train and validation
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
    dataset = DatasetDict({'train': dataset['train'], 'validation': dataset['test']})
    return dataset


def load_local_sts_news_dataset() -> DatasetDict:
    dataset = load_dataset(
        'csv',
        data_files='datasets/STS.news.sr.txt',
        delimiter='\t',
        column_names=['label', 'annotator_1', 'annotator_2', 'annotator_3', 'annotator_4', 'annotator_5', 'text_1', 'text_2'],
        verification_mode='all_checks',
        split='train',
    )
    # cast labels to floats and round to integers (0-5)
    dataset = dataset.map(lambda example: {'label': round(float(example['label']))})
    assert len(dataset.unique('label')) == 6, "Expected 6 unique labels"
    # convert labels to ClassLabel
    class_label = ClassLabel(num_classes=6, names=[str(i) for i in range(6)])
    dataset = dataset.cast_column('label', class_label)
    # split into train and validation
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
    dataset = DatasetDict({'train': dataset['train'], 'validation': dataset['test']})
    return dataset


def load_any_dataset(
        task_name: str,
        sample_share: float | None = None,
        do_sample_as_low_resource: bool = False,
) -> tuple[DatasetDict, str, str | None]:
    if 'glue' in task_name:
        dataset = load_glue_dataset(task_name)
    elif task_name == "senti_comments":
        dataset = load_local_senti_comments_dataset()
    elif task_name == "serbmr_3c":
        dataset = load_local_serbmr_3c_dataset()
    elif task_name == "sts_news":
        dataset = load_local_sts_news_dataset()
    else:
        assert False, f"Unknown task: {task_name}"

    text_1, text_2 = TASK_TO_FIELDS[task_name]

    # sample dataset
    if sample_share:
        if do_sample_as_low_resource:
            dataset["train"] = sample_dataset_as_low_resource(dataset["train"], text_1, sample_share=sample_share)
        else:
            num_samples = int(len(dataset["train"]) * sample_share)
            dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(num_samples))

    return dataset, text_1, text_2


def load_any_metric(task_name: str) -> tuple[Callable[[tuple], dict], str]:
    target_metric_name: str = GLUE_TASK_TO_MAIN_METRIC[task_name]

    if 'glue' in task_name:
        metric_obj = load(*task_name.split("/"), trust_remote_code=True)
    else:
        metric_obj = load(target_metric_name, trust_remote_code=True)

    def compute_metrics(eval_pred: tuple) -> dict:
        logits, labels = eval_pred
        if logits.ndim == 1:
            predictions = logits.squeeze()
        else:
            predictions = logits.argmax(axis=-1)
        if 'glue' in task_name:
            return metric_obj.compute(predictions=predictions, references=labels)
        else:
            return metric_obj.compute(predictions=predictions, references=labels, average='macro')

    return compute_metrics, target_metric_name


@click.command()
@click.option("--model_name", type=str, default="google-bert/bert-base-multilingual-uncased")
@click.option("--task_name", type=str, default="super_glue/boolq")
@click.option("--sample_share", type=float, default=None)
@click.option("--do_sample_as_low_resource", type=bool, default=False, is_flag=True)
@click.option("--learning_rate", type=float, default=5e-5)
@click.option("--batch_size", type=int, default=16)
@click.option("--max_epochs", type=int, default=5)
@click.option("--logging_steps", type=int, default=None)
@click.option("--logging_epochs", type=int, default=None)
@click.option("--aug_type", type=str, default="none")  # none, words-del, words-swap, words-insert, words-sub, chars-del, chars-swap, chars-insert, chars-sub, all
@click.option("--aug_prob", type=float, default=0.25)
@click.option("--aug_words_prob", type=float, default=0.5)
@click.option("--aug_chars_prob", type=float, default=0.25)
def main(
    model_name: str,
    task_name: str,
    sample_share: float | None,
    do_sample_as_low_resource: bool,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    logging_steps: int | None,
    logging_epochs: int | None,
    aug_type: str,
    aug_prob: float,
    aug_words_prob: float,
    aug_chars_prob: float,
) -> None:
    if logging_steps is None and logging_epochs is None:
        logging_epochs = 1
    assert sample_share is None or 0 < sample_share <= 1, "sample_share should be in (0, 1]"
    cleaned_model_name = model_name.replace("/", "-")
    cleaned_task_name = task_name.replace("/", "-")
    results_folder = f"results/{cleaned_model_name}--{cleaned_task_name}--{aug_type}--{sample_share}--{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(results_folder, exist_ok=True)

    print(f"loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    config = AutoConfig.from_pretrained(model_name, num_labels=GLUE_TASK_TO_NUM_LABELS[task_name])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    summary(model)

    print(f"loading dataset: {task_name}")
    dataset, text1_field, text2_field = load_any_dataset(
        task_name, sample_share=sample_share, do_sample_as_low_resource=do_sample_as_low_resource
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"get locale and vocab set from dataset train")
    vocab_set = set()
    for example in dataset["train"]:
        text = example[text1_field].lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        vocab_set.update(text.split())
    locale = "en" if "glue" in task_name else "sr-Latn"
    print(f"locale: {locale}; vocab set size: {len(vocab_set)} ({list(vocab_set)[:10]}...)")

    print(f"creating augmentation pipeline")
    if aug_type == "none":
        augmentation_pipeline = None
    elif aug_type == "all":
        augmentation_pipeline = SelectorAugmenter(
            [
                CharsRandomDeleteAugmenter(aug_words_prob, aug_chars_prob),
                CharsRandomSwapAugmenter(aug_words_prob, aug_chars_prob),
                CharsRandomInsertAugmenter(aug_words_prob, aug_chars_prob, locale=locale),
                CharsRandomSubstituteAugmenter(aug_words_prob, aug_chars_prob, locale=locale),
                WordsRandomDeleteAugmenter(aug_words_prob),
                WordsRandomSwapAugmenter(aug_words_prob),
                WordsRandomInsertAugmenter(aug_words_prob, vocabulary=list(vocab_set)),
                WordsRandomSubstituteAugmenter(aug_words_prob, vocabulary=list(vocab_set)),
            ],
        )
    elif aug_type == "words-all":
        augmentation_pipeline = SequentialAugmenter([
            WordsRandomDeleteAugmenter(aug_words_prob),
            WordsRandomSwapAugmenter(aug_words_prob),
            WordsRandomInsertAugmenter(aug_words_prob, vocabulary=list(vocab_set)),
            WordsRandomSubstituteAugmenter(aug_words_prob, vocabulary=list(vocab_set)),
        ])
    elif aug_type == "words-del":
        augmentation_pipeline = WordsRandomDeleteAugmenter(aug_words_prob)
    elif aug_type == "words-swap":
        augmentation_pipeline = WordsRandomSwapAugmenter(aug_words_prob)
    elif aug_type == "words-insert":
        augmentation_pipeline = WordsRandomInsertAugmenter(aug_words_prob, vocabulary=list(vocab_set))
    elif aug_type == "words-sub":
        augmentation_pipeline = WordsRandomSubstituteAugmenter(aug_words_prob, vocabulary=list(vocab_set))
    elif aug_type == "chars-del":
        augmentation_pipeline = CharsRandomDeleteAugmenter(aug_words_prob, aug_chars_prob)
    elif aug_type == "chars-swap":
        augmentation_pipeline = CharsRandomSwapAugmenter(aug_words_prob, aug_chars_prob)
    elif aug_type == "chars-insert":
        augmentation_pipeline = CharsRandomInsertAugmenter(aug_words_prob, aug_chars_prob, locale=locale)
    elif aug_type == "chars-sub":
        augmentation_pipeline = CharsRandomSubstituteAugmenter(aug_words_prob, aug_chars_prob, locale=locale)
    else:
        assert False, f"Unknown augmentation type: {aug_type}"
    if augmentation_pipeline and aug_prob < 1:
        augmentation_pipeline = ChanceAugmenter(augmentation_pipeline, probability=aug_prob)
    augmentation_pipeline_lambda = lambda text: augmentation_pipeline.augment(text) if augmentation_pipeline and text else text
    print(f"augmentation pipeline: {augmentation_pipeline}")
    print("Some short test text -> ", augmentation_pipeline.augment("Some short test text") if augmentation_pipeline else None)

    train_dataset = AugmentedTokenizedDataset(
        dataset["train"],
        tokenizer,
        text1_field,
        text2_field,
        augmentation_pipeline=augmentation_pipeline_lambda,
    )
    validation_dataset = AugmentedTokenizedDataset(
        dataset["validation"],
        tokenizer,
        text1_field,
        text2_field,
        augmentation_pipeline=None,
    )

    print(f"loading metric: {task_name}")
    compute_metrics, metric_name = load_any_metric(task_name)

    print(f"preparing training arguments")
    training_args = TrainingArguments(
        output_dir=results_folder,
        report_to=[],

        learning_rate=learning_rate,
        lr_scheduler_type='linear',
        weight_decay=0.01,

        auto_find_batch_size=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        warmup_ratio=0.1,

        use_cpu=not IS_CUDA_AVAILABLE,
        fp16=IS_FP16_AVAILABLE,
        fp16_full_eval=IS_FP16_AVAILABLE,

        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # logging_steps=logging_steps,
        # eval_steps=logging_steps,

        # save_steps=logging_steps,
        metric_for_best_model=f"eval_{metric_name}",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        save_only_model=True,

        push_to_hub=False,

        seed=SEED,
    )

    print(f"initializing trainer")
    neptune_callback = NeptuneCallback(
        tags=[model_name, task_name],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), neptune_callback],
    )

    print(f"training model")
    trainer.train()
    run = NeptuneCallback.get_run(trainer)
    run["finetuning/parameters"] = {
        "model_name": model_name,
        "task_name": task_name,
        "sample_share": sample_share,
        "do_sample_as_low_resource": do_sample_as_low_resource,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "logging_steps": logging_steps,
        "aug_type": aug_type,
        "aug_prob": aug_prob,
        "aug_words_prob": aug_words_prob,
        "aug_chars_prob": aug_chars_prob,
    }
    print(f"validating model")
    val_data = trainer.predict(validation_dataset)[-1]
    final_metric = val_data[f"test_{metric_name}"]
    print(val_data)
    print(final_metric)
    run["finetuning/final"] = final_metric


if __name__ == "__main__":
    main()
