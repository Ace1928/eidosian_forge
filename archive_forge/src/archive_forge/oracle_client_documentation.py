import os
import grpc
from keras_tuner.src import protos
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import trial as trial_module
Wraps an `Oracle` on a worker to send requests to the chief.