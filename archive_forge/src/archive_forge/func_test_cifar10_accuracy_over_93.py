import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import autokeras as ak
def test_cifar10_accuracy_over_93(tmp_path):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    clf = ak.ImageClassifier(max_trials=3, directory=tmp_path)
    clf.fit(x_train, y_train, epochs=5)
    accuracy = clf.evaluate(x_test, y_test)[1]
    assert accuracy >= 0.93