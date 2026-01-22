import torch
from xformers.benchmarks.utils import TestCase, bench_functions
from xformers.triton.softmax import log_softmax as triton_log_softmax
from xformers.triton.softmax import softmax as triton_softmax
def to_gbs_fwbw(a, ms):
    return 2 * to_gbs_fw(a, ms)