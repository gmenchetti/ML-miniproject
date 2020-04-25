## CS 412 - Miniproject Track 1

## Description
The task of this project is to build a classification model that is able to predict how empathic a person is. 
More in particular, if a person can be associated to one of the following categories: 
*Very Empathetic*, *Not Very Empathetic*.
## Getting started

### Installation

This project has been built using Python 3.6.

In order to run the program, the following libraries must be installed:
* [scikit-learn](https://scikit-learn.org/stable/documentation.html)
* [numpy](https://docs.scipy.org/doc/)
* [pandas](https://pandas.pydata.org/pandas-docs/stable/)

### Run train
It is possible to run the train code, while inside this folder, executing the following terminal 
line code

```
python train.py <balanced dataset> <one hot encoding>
```

where the last two fields must be replaced with

* balanced dataset: balance (train in the balanced dataset), no_balance (train in the complete dataset)
* one hot encoding: one_hot (train in the one hot encoded dataset), no_one_hot (train in the dataset without one hot encoding)


##### Note

At the end of the training process, the model will be saved in the *models* folder.
Running the train file will override the old models. 

### Run test
It is possible to run the test code, while inside this folder, executing the following terminal 
line code

```
python test.py <balanced dataset> <one hot encoding>
```

where the last two fields must be replaced with

* balanced dataset: balance (test in the balanced dataset), no_balance (test in the complete dataset)
* one hot encoding: one_hot (test in the one hot encoded dataset), no_one_hot (test in the dataset without one hot encoding)


## Notes
The given dataset is already preprocessed and saved in the *n_numpy_ds* folder.
To create a new dataset, run the *create_ds.py* file, indicating whether the new
dataset must be downsampled, if one hot encoding should be applied and setting
the threshold used to apply feature selection.

