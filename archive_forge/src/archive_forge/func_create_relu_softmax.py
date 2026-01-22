import pytest
from thinc.api import (
from thinc.compat import has_tensorflow, has_torch
def create_relu_softmax(width, dropout, nI, nO):
    return chain(clone(Relu(nO=width, dropout=dropout), 2), Softmax(10, width))