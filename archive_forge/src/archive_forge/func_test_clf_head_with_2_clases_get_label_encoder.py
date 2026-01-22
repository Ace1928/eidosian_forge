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
def test_clf_head_with_2_clases_get_label_encoder():
    head = head_module.ClassificationHead(name='a', num_classes=2)
    head._encoded = False
    head._labels = ['a', 'b']
    assert isinstance(head.get_hyper_preprocessors()[-1].preprocessor, preprocessors.LabelEncoder)