import hashlib
import random
import time
from keras_tuner.src import protos
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import metrics_tracking
from keras_tuner.src.engine import stateful
def generate_trial_id():
    s = str(time.time()) + str(random.randint(1, int(10000000.0)))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]