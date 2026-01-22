import torch
import random
import os
import signal
import parlai.utils.distributed as distributed_utils
import parlai.scripts.eval_model as eval_model
from parlai.core.script import ParlaiScript, register_script
def multiprocess_eval(rank, opt, port=61337, rank_offset=0, gpu=None, hostname='localhost'):
    """
    Run a multiprocessing evaluation.

    Invoked by launch_and_eval, not instantiated directly.
    """
    with distributed_utils.distributed_context(rank, opt, port, rank_offset, gpu, hostname) as opt:
        return eval_model.eval_model(opt)