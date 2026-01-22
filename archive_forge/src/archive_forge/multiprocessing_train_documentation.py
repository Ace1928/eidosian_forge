import torch
import random
import os
import signal
import parlai.scripts.train_model as single_train
import parlai.utils.distributed as distributed_utils
from parlai.core.script import ParlaiScript, register_script

    Perform a fork() to many processes.
    