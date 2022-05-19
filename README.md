![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)

# text-classification
Text classification using deep learning model. The purpose is to classify unseen articles into 5 main categories which are Sport,Tech,Business,Entertainment and Politics.

The python scripts uploaded had been tested and run using Spyder(Python 3.8).
<br>The source of the data used for this analysis is:
<br>[Text classification](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv)

### FOLDERS AND FILES
**figures folder**: classification report, tensorboard interface for performance evaluation, model architecture
<br>**__pycache__**: Auto generated file to connect the modules with the training file.
<br>**log_sentiment_analysis**: history of training for tensorboard visualization.
<br>**__init__.py**: initial file to connect classes and functions with training file.
<br>**ohe.pkl**: pickle file stored one hot encoder trained
<br>**sentiment_analysis_h5**: saved trained model
<br>**textdoc_modules.py**: all the classes and functions created to ease the training process
<br>**textdoc_train.py**: python script for model training
<br>**tokenizer_sentiment.json**: json file stored tokenizer trained

### MODEL




Embedding layer is used to 
Bidirectional layer of LSTM can
tanh activation function in LSTM
Dropout layer prevented overfitting


The performance of the model is viewed from tensorboard. In order to access to the tensorboard, you will need to follow the steps below:
Open anaconda prompt> conda activate (environment name) > tensorboard --logdir (path pointed to relevant log file)
Tensorboard show performance of 

### IMPROVEMENTS/SUGGESTIONS
1. Training deep learning model required more times because it analyzed based on large amount of words. Use google colab to train the model if the capacity of your device is not sufficient.
2. Can try to explore more ways of data cleaning in order to ensure the input loading to the deep learning model is sufficient with adequate amount of data.



Thanks for reading.
