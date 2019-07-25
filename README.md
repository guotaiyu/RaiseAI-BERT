## Project layout
```
Raise-BERT/
├── README.md
├── bert
│   ├── inputprocessor.py
│   ├── predictor.py
│   ├── run_engine.py
│   ├── tokenization.py
│   ├── vocab.txt

```

## Bring up service
```
cd bert
```

In ```bert``` directory, run
```
python run_engine.py
```

Open another terminal window and run
```
curl -i "http://127.0.0.1:5000?text1=Sally+is+a+male&text2=Sally+is+a+man"
```
