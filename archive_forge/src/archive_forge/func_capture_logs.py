import torch
from torch.utils._pytree import tree_map
from typing import Iterator, List, Optional
import logging
import contextlib
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary
import functools
from torch._C._profiler import gather_traceback, symbolize_tracebacks
@contextlib.contextmanager
def capture_logs(is_mode=False, python_tb=False, script_tb=False, cpp_tb=False) -> Iterator[List[str]]:
    collect_traceback = python_tb or script_tb or cpp_tb
    logger = logging.getLogger('LoggingTensor')
    log_list: List[str] = []
    tracebacks_list: List[str] = []
    handler = LoggingTensorHandler(log_list, with_type=True, use_shortid_for_all_tensors=is_mode, tracebacks_list=tracebacks_list if collect_traceback else None)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if collect_traceback:
        logger.addFilter(GatherTraceback(python=python_tb, script=script_tb, cpp=cpp_tb))
    try:
        if collect_traceback:
            yield (log_list, tracebacks_list)
        else:
            yield log_list
    finally:
        symbolized_tracebacks = symbolize_tracebacks(tracebacks_list)
        tracebacks_list.clear()
        tracebacks_list.extend(symbolized_tracebacks)
        logger.removeHandler(handler)