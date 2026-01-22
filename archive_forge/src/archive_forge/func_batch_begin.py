import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
def batch_begin(self, estimator, *args, **kwargs):
    if isinstance(self.log_interval, int):
        self.batch_start = time.time()