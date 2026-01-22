from typing import Optional
from tensorflow import keras
from autokeras.engine import io_hypermodel
from autokeras.utils import types
def deserialize_metrics(metrics):
    deserialized = []
    for metric in metrics:
        if isinstance(metric, list):
            deserialized.append(metric[0])
        else:
            deserialized.append(keras.metrics.deserialize(metric))
    return deserialized