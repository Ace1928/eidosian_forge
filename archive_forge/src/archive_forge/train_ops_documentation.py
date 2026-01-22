import logging
import numpy as np
import math
from typing import Dict
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.sgd import do_minibatch_sgd
from ray.util import log_once
Multi-GPU version of train_one_step.

    Uses the policies' `load_batch_into_buffer` and `learn_on_loaded_batch` methods
    to be more efficient wrt CPU/GPU data transfers. For example, when doing multiple
    passes through a train batch (e.g. for PPO) using `config.num_sgd_iter`, the
    actual train batch is only split once and loaded once into the GPU(s).

    .. testcode::
        :skipif: True

        from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
        algo = [...]
        train_batch = synchronous_parallel_sample(algo.workers)
        # This trains the policy on one batch.
        print(multi_gpu_train_one_step(algo, train_batch)))

    .. testoutput::

        {"default_policy": ...}

    Updates the NUM_ENV_STEPS_TRAINED and NUM_AGENT_STEPS_TRAINED counters as well as
    the LOAD_BATCH_TIMER and LEARN_ON_BATCH_TIMER timers of the Algorithm instance.
    