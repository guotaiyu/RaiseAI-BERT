## Project layout
```
Raise-BERT/
├── README.md
├── bert/
│   ├── __init__.py
│   ├── inputprocessor.py
│   ├── predictor.py
│   ├── run_engine.py
│   └── tokenization.py
└── model/
    └── 1/                  # version no.
        ├── vocab.txt
        ├── saved_model.pb
        └── variables/
            ├── variables.data-00000-of-00001
            └── variables.index
```

## Serve the fine-tuned bert model using tensorflow serving
Follow the instruction [here](https://www.tensorflow.org/tfx/serving/setup) to install ```tensorflow-model-server```

Download fine-tuned bert model from [here](https://drive.google.com/drive/folders/1Lsieg1PMLYZKjbncQfnwaZqcd04_Sev_?usp=sharing) and save it to ```model``` directory


Serve the model:
```
tensorflow_model_server --rest_api_port=9000 --model_name=bert --model_base_path=Raise-BERT/model
```
```
****
... ... ...
... ... ...
I tensorflow_serving/model_servers/server.cc:344] Exporting HTTP/REST API at:localhost:9000 ...
****
```

## Bring up Flask application
Open a separate terminal and in root directory, run:
```
./run_engine
```
```
****
 * Serving Flask app "run_engine" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: on
WARNING: Logging before flag parsing goes to stderr.
W0725 12:26:22.327455 140446491150144 _internal.py:88]  * Debugger is active!
****
```

Open another terminal and run:
```
curl -i "http://127.0.0.1:5000?text1=Sally+is+a+male&text2=Sally+is+a+man"
```
```
****
HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 99
Server: Werkzeug/0.14.1 Python/3.7.3
Date: Thu, 25 Jul 2019 16:28:05 GMT

{
    "contradiction": 0.0107518993,
    "entailment": 0.987416089,
    "neutral": 0.00183196843
}
****
```
