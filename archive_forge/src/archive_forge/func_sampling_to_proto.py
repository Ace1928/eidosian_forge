import math
from keras_tuner.src import protos
def sampling_to_proto(sampling):
    if sampling == 'linear':
        return protos.get_proto().Sampling.LINEAR
    if sampling == 'log':
        return protos.get_proto().Sampling.LOG
    if sampling == 'reverse_log':
        return protos.get_proto().Sampling.REVERSE_LOG
    raise ValueError(f"Expected sampling to be 'linear', 'log', or 'reverse_log'. Received: '{sampling}'.")