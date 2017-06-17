# Machine Learning Engineer Nanodegree
# Capstone Project
## Project: Predict Russian Housing Market Price

### Install

This project requires **Python 3.6** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/#installation)
- [imputer - Knn](https://github.com/bwanglzu/Imputer.py)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

It is recommended to install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

### Run

In a terminal or command window, navigate to the top-level project directory `Kaggle_Competition_RussiaEconomy/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Capstone_Report.ipynb
```  
or
```bash
jupyter notebook Capstone_Report.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The dataset consists of 3 files,
- `macro.csv` : 2484 entries with 100 features describing the value of different macro economic indicator with Timestamp
- `train.csv` : 30471 entries with 291 features & 1 outcome. It describes the individual situtation of individual housing that buy/sell in certain Timestamp. With 1 outcome stating the real trading price of that specific house
- `test.csv` : 7662 entries with simliar structure as train.csv except there is no outcome to validate. It can only be tested in Kaggle Portal so we will focus on the CV score of train.csv but we will also upload our result to Kaggle to test the exact when it handles external dataa

This dataset is internal data provided by Sberbank for the sake of competition purpose. You can also get the original dataset hosted on [Kaggle - Competition](https://www.kaggle.com/c/sberbank-russian-housing-market/data).

### Features

You can get the detail description of features in ./data_dictionary.txt
