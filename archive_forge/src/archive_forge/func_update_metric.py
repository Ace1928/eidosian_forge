import json
from collections import deque
from numbers import Number
from typing import Tuple, Optional
from ray.train._internal.checkpoint_manager import _CheckpointManager
from ray.tune.utils.serialization import TuneFunctionEncoder, TuneFunctionDecoder
def update_metric(self, metric: str, value: Number, step: Optional[int]=1):
    if metric not in self.metric_analysis:
        self.metric_analysis[metric] = {'max': value, 'min': value, 'avg': value, 'last': value}
        self.metric_n_steps[metric] = {}
        for n in self._n_steps:
            key = 'last-{:d}-avg'.format(n)
            self.metric_analysis[metric][key] = value
            self.metric_n_steps[metric][str(n)] = deque([value], maxlen=n)
    else:
        step = step or 1
        self.metric_analysis[metric]['max'] = max(value, self.metric_analysis[metric]['max'])
        self.metric_analysis[metric]['min'] = min(value, self.metric_analysis[metric]['min'])
        self.metric_analysis[metric]['avg'] = 1 / step * (value + (step - 1) * self.metric_analysis[metric]['avg'])
        self.metric_analysis[metric]['last'] = value
        for n in self._n_steps:
            key = 'last-{:d}-avg'.format(n)
            self.metric_n_steps[metric][str(n)].append(value)
            self.metric_analysis[metric][key] = sum(self.metric_n_steps[metric][str(n)]) / len(self.metric_n_steps[metric][str(n)])
    self.invalidate_cache()