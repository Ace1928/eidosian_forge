import random
import pytest
from thinc.api import (
def evaluate_tagger(model, dev_X, dev_Y, batch_size):
    correct = 0.0
    total = 0.0
    for i in range(0, len(dev_X), batch_size):
        Yh = model.predict(dev_X[i:i + batch_size])
        Y = dev_Y[i:i + batch_size]
        for j in range(len(Yh)):
            correct += (Yh[j].argmax(axis=1) == Y[j].argmax(axis=1)).sum()
            total += Yh[j].shape[0]
    return correct / total