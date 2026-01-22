import copy
import queue
import threading
from typing import Dict, Optional
from ray.util.timer import _Timer
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.execution.minibatch_buffer import MinibatchBuffer
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder, LEARNER_INFO
from ray.rllib.utils.metrics.window_stat import WindowStat
from ray.util.iter import _NextValueNotReady
def timer_to_ms(timer):
    return round(1000 * timer.mean, 3)