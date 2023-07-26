import tensorflow as tf

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import datetime

import matplotlib.pyplot as plt



def mae(y_test, y_pred):
  """
  Calculuates mean absolute error between y_test and y_preds.
  """
  return tf.metrics.mean_absolute_error(y_test,
                                        y_pred)
  
def mse(y_test, y_pred):
  """
  Calculates mean squared error between y_test and y_preds.
  """
  return tf.metrics.mean_squared_error(y_test,
                                       y_pred)



def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = y_preds):
  """
  Plots training data, test data and predictions
  """
  plt.figure(figsize=(8,5))
  #Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label = "Training data")
  #Plot testing data in green 
  plt.scatter(test_data, test_labels, c="g", label = "Testing data")
  # Plot predictions in red
  plt.scatter(test_data, predictions , c="r", label = "Predictions ")
  # Show the legend 
  plt.legend();
