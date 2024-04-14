# UdS 2024: Final project for NLP for Low-Resource Languages course
## Data Augmentation limits in Low-Resource environment: A Case Study with Serbian


## How Install

To run the training script, you need to have Python 3.12 and the required packages installed. 
```bash
pip install -r requirements.txt
```

Additionally, you need to fill .env file with your Neptune.ai `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN` to log the experiments.

## Main Libraries used
* `transformers` for obtaining the checkpoints, training loop and evaluation
* `datasets` for loading the Super GLUE datasets
* `fast-aug` - our [custom library](https://github.com/k4black/fast-aug) for random data augmentation - written on rust with python bindings
* `neptune` for logging the experiments ([runs available](https://app.neptune.ai/k4black/uds-coli/runs/table?viewId=9b9b8004-c615-4fd7-a04f-e4b91755add0&detailsTab=dashboard&dashboardId=9b9b8193-6b6a-4bdb-a824-c1f45450129b&shortId=US1-72&dash=charts&type=run))


## Run

To get all the available options, run:
```bash
python main.py --help
```

For example, to train the roberta-base model on the CB task with words substitution augmentation, run:
```bash
python main.py --task_name super_glue/cb --model_name roberta-base --augmentation words-sub
```
