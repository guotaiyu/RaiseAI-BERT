## Project layout
```
Raise-BERT/
├── README.md
├── bert
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   ├── README.md
│   ├── __init__.py
│   ├── create_pretraining_data.py
│   ├── dataprocessor.py
│   ├── engine.py
│   ├── extract_features.py
│   ├── inputprocessor.py
│   ├── modeling.py
│   ├── modeling_test.py
│   ├── multilingual.md
│   ├── optimization.py
│   ├── optimization_test.py
│   ├── prediction.py
│   ├── requirements.txt
│   ├── run_classifier.py
│   ├── run_classifier_with_tfhub.py
│   ├── run_pretraining.py
│   ├── run_squad.py
│   ├── sample_text.txt
│   ├── tokenization.py
│   └── tokenization_test.py
├── data
│   └── MNLI
│       ├── README.txt
│       ├── dev_matched.tsv
│       ├── dev_mismatched.tsv
│       ├── original
│       │   ├── multinli_1.0_dev_matched.jsonl
│       │   ├── multinli_1.0_dev_matched.txt
│       │   ├── multinli_1.0_dev_mismatched.jsonl
│       │   ├── multinli_1.0_dev_mismatched.txt
│       │   ├── multinli_1.0_train.jsonl
│       │   └── multinli_1.0_train.txt
│       ├── test_matched.tsv
│       ├── test_mismatched.tsv
│       └── train.tsv
├── model
│   ├── model.ckpt-36815.data-00000-of-00001
│   ├── model.ckpt-36815.index
│   ├── model.ckpt-36815.meta
│   └── uncased_L-12_H-768_A-12
│       ├── bert_config.json
│       ├── bert_model.ckpt.data-00000-of-00001
│       ├── bert_model.ckpt.index
│       ├── bert_model.ckpt.meta
│       ├── pytorch_model.bin
│       └── vocab.txt
└── output
    ├── checkpoint
    ├── eval
    │   └── events.out.tfevents.1563419342.raise-ai-dev
    ├── eval.tf_record
    ├── eval_results.txt
    ├── events.out.tfevents.1563415727.raise-ai-dev
    ├── events.out.tfevents.1563416837.raise-ai-dev
    ├── graph.pbtxt
    ├── temp
    └── train.tf_record
```

## Bring up service
In the root directory, run
```
python bert/engine.py
```

Open another terminal window and run
```
curl -i "http://127.0.0.1:5000?text1=Sally+is+a+male&text2=Sally+is+a+man"
```
