import threading
from keras.src import backend
from keras.src.api_export import keras_export
def set_global_attribute(name, value):
    setattr(GLOBAL_STATE_TRACKER, name, value)