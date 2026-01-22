from typing import Optional
from tensorflow import keras
from autokeras.engine import io_hypermodel
from autokeras.utils import types
def serialize_loss(loss):
    if isinstance(loss, str):
        return [loss]
    return keras.losses.serialize(loss)