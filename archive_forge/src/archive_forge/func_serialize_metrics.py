from typing import Optional
from tensorflow import keras
from autokeras.engine import io_hypermodel
from autokeras.utils import types
def serialize_metrics(metrics):
    serialized = []
    for metric in metrics:
        if isinstance(metric, str):
            serialized.append([metric])
        else:
            serialized.append(keras.metrics.serialize(metric))
    return serialized