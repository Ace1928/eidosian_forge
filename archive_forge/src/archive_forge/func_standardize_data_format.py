import json
import os
from keras.src.api_export import keras_export
def standardize_data_format(data_format):
    if data_format is None:
        return image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError(f"The `data_format` argument must be one of {{'channels_first', 'channels_last'}}. Received: data_format={data_format}")
    return data_format