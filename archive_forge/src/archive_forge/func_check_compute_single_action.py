from collections import Counter
import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Tuple as GymTuple
import inspect
import logging
import numpy as np
import os
import pprint
import random
import re
import time
import tree  # pip install dm_tree
from typing import (
import yaml
import ray
from ray import air, tune
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter, run_experiments
def check_compute_single_action(algorithm, include_state=False, include_prev_action_reward=False):
    """Tests different combinations of args for algorithm.compute_single_action.

    Args:
        algorithm: The Algorithm object to test.
        include_state: Whether to include the initial state of the Policy's
            Model in the `compute_single_action` call.
        include_prev_action_reward: Whether to include the prev-action and
            -reward in the `compute_single_action` call.

    Raises:
        ValueError: If anything unexpected happens.
    """
    from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
    pid = DEFAULT_POLICY_ID
    try:
        pid = next(iter(algorithm.workers.local_worker().get_policies_to_train()))
        pol = algorithm.get_policy(pid)
    except AttributeError:
        pol = algorithm.policy
    model = pol.model
    action_space = pol.action_space

    def _test(what, method_to_test, obs_space, full_fetch, explore, timestep, unsquash, clip):
        call_kwargs = {}
        if what is algorithm:
            call_kwargs['full_fetch'] = full_fetch
            call_kwargs['policy_id'] = pid
        obs = obs_space.sample()
        if isinstance(obs_space, Box):
            obs = np.clip(obs, -1.0, 1.0)
        state_in = None
        if include_state:
            state_in = model.get_initial_state()
            if not state_in:
                state_in = []
                i = 0
                while f'state_in_{i}' in model.view_requirements:
                    state_in.append(model.view_requirements[f'state_in_{i}'].space.sample())
                    i += 1
        action_in = action_space.sample() if include_prev_action_reward else None
        reward_in = 1.0 if include_prev_action_reward else None
        if method_to_test == 'input_dict':
            assert what is pol
            input_dict = {SampleBatch.OBS: obs}
            if include_prev_action_reward:
                input_dict[SampleBatch.PREV_ACTIONS] = action_in
                input_dict[SampleBatch.PREV_REWARDS] = reward_in
            if state_in:
                if what.config.get('_enable_new_api_stack', False):
                    input_dict['state_in'] = state_in
                else:
                    for i, s in enumerate(state_in):
                        input_dict[f'state_in_{i}'] = s
            input_dict_batched = SampleBatch(tree.map_structure(lambda s: np.expand_dims(s, 0), input_dict))
            action = pol.compute_actions_from_input_dict(input_dict=input_dict_batched, explore=explore, timestep=timestep, **call_kwargs)
            if isinstance(action[0], list):
                action = (np.array(action[0]), action[1], action[2])
            action = tree.map_structure(lambda s: s[0], action)
            try:
                action2 = pol.compute_single_action(input_dict=input_dict, explore=explore, timestep=timestep, **call_kwargs)
                if not explore and (not pol.config.get('noisy')):
                    check(action, action2)
            except TypeError:
                pass
        else:
            action = what.compute_single_action(obs, state_in, prev_action=action_in, prev_reward=reward_in, explore=explore, timestep=timestep, unsquash_action=unsquash, clip_action=clip, **call_kwargs)
        state_out = None
        if state_in or full_fetch or what is pol:
            action, state_out, _ = action
        if state_out:
            for si, so in zip(tree.flatten(state_in), tree.flatten(state_out)):
                if tf.is_tensor(si):
                    si_shape = si.shape.as_list()
                else:
                    si_shape = list(si.shape)
                check(si_shape, so.shape)
        if unsquash is None:
            unsquash = what.config['normalize_actions']
        if clip is None:
            clip = what.config['clip_actions']
        if method_to_test == 'single' and what == algorithm:
            if not action_space.contains(action) and (clip or unsquash or (not isinstance(action_space, Box))):
                raise ValueError(f"Returned action ({action}) of algorithm/policy {what} not in Env's action_space {action_space}")
            if isinstance(action_space, Box) and (not unsquash) and what.config.get('normalize_actions') and np.any(np.abs(action) > 15.0):
                raise ValueError(f'Returned action ({action}) of algorithm/policy {what} should be in normalized space, but seems too large/small for that!')
    for what in [pol, algorithm]:
        if what is algorithm:
            worker_set = getattr(algorithm, 'workers', None)
            assert worker_set
            if not worker_set.local_worker():
                obs_space = algorithm.get_policy(pid).observation_space
            else:
                obs_space = worker_set.local_worker().for_policy(lambda p: p.observation_space, policy_id=pid)
            obs_space = getattr(obs_space, 'original_space', obs_space)
        else:
            obs_space = pol.observation_space
        for method_to_test in ['single'] + (['input_dict'] if what is pol else []):
            for explore in [True, False]:
                for full_fetch in [False, True] if what is algorithm else [False]:
                    timestep = random.randint(0, 100000)
                    for unsquash in [True, False, None]:
                        for clip in [False] if unsquash else [True, False, None]:
                            print('-' * 80)
                            print(f'what={what}')
                            print(f'method_to_test={method_to_test}')
                            print(f'explore={explore}')
                            print(f'full_fetch={full_fetch}')
                            print(f'unsquash={unsquash}')
                            print(f'clip={clip}')
                            _test(what, method_to_test, obs_space, full_fetch, explore, timestep, unsquash, clip)