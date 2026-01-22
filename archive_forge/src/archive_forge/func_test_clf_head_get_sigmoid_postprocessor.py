import keras_tuner
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
import autokeras as ak
from autokeras import hyper_preprocessors
from autokeras import nodes as input_module
from autokeras import preprocessors
from autokeras import test_utils
from autokeras.blocks import heads as head_module
def test_clf_head_get_sigmoid_postprocessor():
    head = head_module.ClassificationHead(name='a', multi_label=True)
    head._encoded = True
    head._encoded_for_sigmoid = True
    assert isinstance(head.get_hyper_preprocessors()[0].preprocessor, preprocessors.SigmoidPostprocessor)