import re
import warnings
import keras_tuner
import tensorflow as tf
from packaging.version import parse
from tensorflow import nest
def contain_instance(instance_list, instance_type):
    return any([isinstance(instance, instance_type) for instance in instance_list])