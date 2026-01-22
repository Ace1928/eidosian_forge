import torch
import random
import os
import signal
import parlai.scripts.train_model as single_train
import parlai.utils.distributed as distributed_utils
from parlai.core.script import ParlaiScript, register_script
def multiprocess_train(rank, opt, port=61337, rank_offset=0, gpu=None, hostname='localhost'):
    with distributed_utils.distributed_context(rank, opt, port, rank_offset, gpu, hostname) as opt:
        return single_train.TrainLoop(opt).train()