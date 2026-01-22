from typing import Dict, Any
from abc import abstractmethod
from itertools import islice
import os
from tqdm import tqdm
import random
import torch
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import warn_once
from parlai.utils.torch import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.core.metrics import AverageMetric
import parlai.utils.logging as logging
def set_interactive_mode(self, mode, shared=False):
    """
        Set interactive mode defaults.

        In interactive mode, we set `ignore_bad_candidates` to True.
        Additionally, we change the `eval_candidates` to the option
        specified in `--interactive-candidates`, which defaults to False.

        Interactive mode possibly changes the fixed candidates path if it
        does not exist, automatically creating a candidates file from the
        specified task.
        """
    super().set_interactive_mode(mode, shared)
    if not mode:
        return
    self.eval_candidates = self.opt.get('interactive_candidates', 'fixed')
    if self.eval_candidates == 'fixed':
        if self.fixed_candidates_path is None or self.fixed_candidates_path == '':
            path = self.get_task_candidates_path()
            if path:
                if not shared:
                    logging.info(f'Setting fixed_candidates path to: {path}')
                self.fixed_candidates_path = path
    self.ignore_bad_candidates = True
    return