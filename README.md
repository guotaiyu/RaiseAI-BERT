## Project layout
```
Raise-BERT/
├── README.md
└── bert/
│   ├── inputprocessor.py
│   ├── predictor.py
│   ├── run_engine.py
│   ├── tokenization.py
│   └── vocab.txt
└── model/
    └── 1/ (version #)
        ├── saved_model.pb
        └── variables/
            ├── variables.data-00000-of-00001
            └── variables.index
```

## Serve the fine-tuned bert model using tensorflow serving
Follow the instruction [here](https://www.tensorflow.org/tfx/serving/setup) to install ```tensorflow-model-server```

Download fine-tuned bert model from [here](https://drive.google.com/drive/folders/1Lsieg1PMLYZKjbncQfnwaZqcd04_Sev_?usp=sharing) and save it to ```model``` directory

Show signatures of the model:
```
saved_model_cli show --dir Raise-BERT/model/1 --all
```

Signature Details:
```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_ids'] tensor_info:
        dtype: DT_INT32
        shape: (1, 128)
        name: input_ids:0
    inputs['input_mask'] tensor_info:
        dtype: DT_INT32
        shape: (1, 128)
        name: input_mask:0
    inputs['label_ids'] tensor_info:
        dtype: DT_INT32
        shape: (1)
        name: label_ids:0
    inputs['segment_ids'] tensor_info:
        dtype: DT_INT32
        shape: (1, 128)
        name: segment_ids:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['probabilities'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 3)
        name: loss/Softmax:0
  Method name is: tensorflow/serving/predict
```

Serve the model:
```
tensorflow_model_server --rest_api_port=9000 --model_name=bert --model_base_path=Raise-BERT/model
```

## Bring up Flask application
Open a separate terminal and in ```bert``` directory, run:
```
python run_engine.py
```

Open another terminal and run:
```
curl -i "http://127.0.0.1:5000?text1=Sally+is+a+male&text2=Sally+is+a+man"
```
