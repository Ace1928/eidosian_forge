import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
def batch_end(self, estimator, *args, **kwargs):
    loss = kwargs['loss']
    batch_size = 0
    if not isinstance(loss, list):
        loss = [loss]
    if isinstance(loss, list):
        for l in loss:
            batch_size += l.shape[0]
    estimator.trainer.step(batch_size)