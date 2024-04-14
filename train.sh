nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


augmentation_chances="0.1 0.3 0.5 0.7 0.9"

for augmentation_chance in $augmentation_chances; do
    echo "-------------------"
    echo "Running with augmentation_chance: $augmentation_chance"
    echo "-------------------"

    python main.py --task_name=glue/mrpc        --model_name=roberta-base       --learning_rate=2e-5 --batch_size=16 --max_epochs=8 --aug_type=words-all --aug_prob=$augmentation_chance --aug_words_prob=0.2 --aug_chars_prob=0.2
    python main.py --task_name=super_glue/boolq --model_name=roberta-base       --learning_rate=2e-5 --batch_size=16 --max_epochs=8 --aug_type=words-all --aug_prob=$augmentation_chance --aug_words_prob=0.2 --aug_chars_prob=0.2
    python main.py --task_name=super_glue/cb    --model_name=roberta-base       --learning_rate=5e-5 --batch_size=16 --max_epochs=8 --aug_type=words-all --aug_prob=$augmentation_chance --aug_words_prob=0.2 --aug_chars_prob=0.2
    python main.py --task_name=senti_comments   --model_name=Andrija/SRoBERTa-F --learning_rate=5e-5 --batch_size=16 --max_epochs=8 --aug_type=words-all --aug_prob=$augmentation_chance --aug_words_prob=0.2 --aug_chars_prob=0.2
    python main.py --task_name=serbmr_3c        --model_name=Andrija/SRoBERTa-F --learning_rate=2e-5 --batch_size=16 --max_epochs=8 --aug_type=words-all --aug_prob=$augmentation_chance --aug_words_prob=0.2 --aug_chars_prob=0.2
    python main.py --task_name=sts_news         --model_name=Andrija/SRoBERTa-F --learning_rate=5e-5 --batch_size=16 --max_epochs=8 --aug_type=words-all --aug_prob=$augmentation_chance --aug_words_prob=0.2 --aug_chars_prob=0.2
done
