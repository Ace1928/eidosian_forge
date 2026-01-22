import os
import time
from concurrent import futures
import grpc
from keras_tuner.src import protos
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine.oracle import synchronized
@synchronized
def exit_chief(oracle):
    return len(oracle.ongoing_trials) == 0 and len(oracle.tuner_ids) == 0