# Contradictory Phrases Application
*** Authors: Keegan Philips, Sara Galego, Rania Bader, Jackie Zuker
## Goal
The goal of this application is to productionize and serve up the ability for the user to input two phrases in any language, and to return insights on whether the phrases contradict each other. 

### Set up
Everything, including data prep, model training and predictions, lives in the single file 
[train_and_predict.py](contradiction_app/train_and_predict.py). Corresponding tests are in 
[test_train_and_predict](contradiction_app/tests/test_train_and_predict.py).

Details on the data raw data are below:


## Setup
To run tests and to train and predict the model via Docker you can use the following commands: 
```bash
docker-compose -f docker-compose.yml -p contradiction_app build
docker-compose -f docker-compose.yml -p contradiction_app up
docker-compose -f docker-compose.yml -p contradiction_app down
```

To run things outside of Docker you'll first need to build your environment. Using Python 3.9 install the required packages
via:
```bash
pip install -r requirements.txt
```
Then to run tests:
```bash
cd contradiction_app
python -m pytest contradiction_app/tests/test_train_and_predict.py
```

Then to train the model and generate predictions:
```bash
python contradiction_app/train_and_predict.py
```

