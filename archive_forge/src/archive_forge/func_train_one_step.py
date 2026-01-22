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
@DeveloperAPI
def train_one_step(algorithm, train_batch, policies_to_train=None) -> Dict:
    """Function that improves the all policies in `train_batch` on the local worker.

    .. testcode::
        :skipif: True

        from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
        algo = [...]
        train_batch = synchronous_parallel_sample(algo.workers)
        # This trains the policy on one batch.
        print(train_one_step(algo, train_batch)))

    .. testoutput::

        {"default_policy": ...}

    Updates the NUM_ENV_STEPS_TRAINED and NUM_AGENT_STEPS_TRAINED counters as well as
    the LEARN_ON_BATCH_TIMER timer of the `algorithm` object.
    """
    config = algorithm.config
    workers = algorithm.workers
    local_worker = workers.local_worker()
    num_sgd_iter = config.get('num_sgd_iter', 1)
    sgd_minibatch_size = config.get('sgd_minibatch_size', 0)
    learn_timer = algorithm._timers[LEARN_ON_BATCH_TIMER]
    with learn_timer:
        if num_sgd_iter > 1 or sgd_minibatch_size > 0:
            info = do_minibatch_sgd(train_batch, {pid: local_worker.get_policy(pid) for pid in policies_to_train or local_worker.get_policies_to_train(train_batch)}, local_worker, num_sgd_iter, sgd_minibatch_size, [])
        else:
            info = local_worker.learn_on_batch(train_batch)
    learn_timer.push_units_processed(train_batch.count)
    algorithm._counters[NUM_ENV_STEPS_TRAINED] += train_batch.count
    algorithm._counters[NUM_AGENT_STEPS_TRAINED] += train_batch.agent_steps()
    if algorithm.reward_estimators:
        info[DEFAULT_POLICY_ID]['off_policy_estimation'] = {}
        for name, estimator in algorithm.reward_estimators.items():
            info[DEFAULT_POLICY_ID]['off_policy_estimation'][name] = estimator.train(train_batch)
    return info