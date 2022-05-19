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
<br>**log_sentiment_analysis folder**: history of training for tensorboard visualization.
<br>**__pycache__**: Auto generated file to connect the modules with the training file.
<br>**__init__.py**: initial file to connect classes and functions with training file.
<br>**ohe.pkl**: pickle file stored one hot encoder trained
<br>**sentiment_analysis_h5**: saved trained model
<br>**textdoc_modules.py**: all the classes and functions created to ease the training process
<br>**textdoc_train.py**: python script for model training
<br>**tokenizer_sentiment.json**: json file stored tokenizer trained

### MODEL
Figure below show the architecture of the model.

![Image](https://github.com/innju/text-classification/blob/main/figures/model.png)

Embedding layer is used to fasten the training process. LSTM layer kept all the relevant information as it passes through every layer. Bidirectional layer of LSTM also is used to make the training process faster. It capable in utilizing information from both sides.tanh activation function is used to overcome vanishing gradient.Dropout layer is used to prevent overfitting. The value is only 0.2 to prevent too many information loss during data training.

![Image](https://github.com/innju/text-classification/blob/main/figures/textdoc_classification_report.png)

Classification report shows the accuracy of the model is 0.90 which is equivalent to 90.0%.It can accurately predict for all the categories.

![Image](https://github.com/innju/text-classification/blob/main/figures/textdoc_tensorboard.png)

The performance of the model is viewed from tensorboard. In order to access to the tensorboard, you will need to follow the steps below:
Open anaconda prompt> conda activate (environment name) > tensorboard --logdir (path pointed to relevant log file)
<br>Tensorboard show performance of model through the epoch loss and epoch accuracy. The training data (orange line) gained higher value for epoch accuracy and lower value for epoch loss compared to the validate (blue line). This means the data actually trained well with the training data only. Suspecting there is overfitting occured here because the epoch loss for validate is much more higher, with the value of 0.54. Possible solution is to reduce the complexity of the model.  


### IMPROVEMENTS/SUGGESTIONS
1. Training deep learning model required more times because it analyzed based on large amount of words. Use google colab to train the model if the capacity of your device is not sufficient.
2. Can try to explore more ways of data cleaning in order to ensure the input loading to the deep learning model is sufficient with adequate amount of data.
3. I gained high accuracy when trained with less epochs and higher accuracy when trained with higher epochs. It is always adviseable to train for lower and higher epochs to see if there is any changes in performance. 


Thanks for reading.
