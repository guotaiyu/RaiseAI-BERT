## Project layout
```
RaiseAI-BERT/
├── README.md
├── start                   # executable
├── bert/
│   ├── __init__.py
│   ├── inputprocessor.py
│   ├── predictor.py
│   └── tokenization.py
└── model/
    ├── vocab.txt
    └── 1/                  # version no.
        ├── saved_model.pb
        └── variables/
            ├── variables.data-00000-of-00001
            └── variables.index
```

## Prerequisite: Serve the fine-tuned BERT model using Tensorflow Serving
### Linux/MacOS
Docker images are the default choice for using Tensorflow ModelServer. Assuming Docker has been installed, we can pull the latest Tensorflow Serving Docker image by running: 
```
docker pull tensorflow/serving
```
Then, in the root directory, we can run the Docker container for Tensorflow ModelServer using:
```
docker run -p $(PORT):8501 --mount type=bind,source=$(pwd)/model/,target=/models/bert -e MODEL_NAME=bert -t tensorflow/serving
```
where `PORT` is the port number we intend to expose for our entailment service. By default, we use `PORT=9000`. Note that if you want to use a different port other than `9000`, you'll also have to modify the variable `tf_serving_url` found in `bert/predictor.py`. Also, note that port `8501` is the port exposed by the Tensorflow ModelServer REST API. 

For more details, refer to the instructions [here](https://www.tensorflow.org/tfx/serving/docker). 

### Ubuntu
Follow the instruction [here](https://www.tensorflow.org/tfx/serving/setup) to install ```tensorflow-model-server```. This is the command that brings up Tensorflow ModelServer.

Download fine-tuned bert model from [here](https://drive.google.com/drive/folders/1Lsieg1PMLYZKjbncQfnwaZqcd04_Sev_?usp=sharing) and save it to ```model``` directory


Serve the model:
```
tensorflow_model_server --rest_api_port=$(PORT) --model_name=bert --model_base_path=RaiseAI-BERT/model
```

## Bring up Entailment Service
To bring up our entailment service, open another terminal window. Then, go to the root directory and run:
```
./start
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

On a separate terminal window, run:
```
curl -i "http://127.0.0.1:5000?text=Sally+is+a+male&hypo=Sally+is+a+man"
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
