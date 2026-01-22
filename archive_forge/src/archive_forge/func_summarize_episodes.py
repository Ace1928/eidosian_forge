import collections
import logging
import numpy as np
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import GradInfoDict, LearnerStatsDict, ResultDict
@DeveloperAPI
def summarize_episodes(episodes: List[RolloutMetrics], new_episodes: List[RolloutMetrics]=None, keep_custom_metrics: bool=False) -> ResultDict:
    """Summarizes a set of episode metrics tuples.

    Args:
        episodes: List of most recent n episodes. This may include historical ones
            (not newly collected in this iteration) in order to achieve the size of
            the smoothing window.
        new_episodes: All the episodes that were completed in this iteration.
        keep_custom_metrics: Whether to keep custom metrics in the result dict as
            they are (True) or to aggregate them (False).

    Returns:
        A result dict of metrics.
    """
    if new_episodes is None:
        new_episodes = episodes
    episode_rewards = []
    episode_lengths = []
    policy_rewards = collections.defaultdict(list)
    custom_metrics = collections.defaultdict(list)
    perf_stats = collections.defaultdict(list)
    hist_stats = collections.defaultdict(list)
    episode_media = collections.defaultdict(list)
    connector_metrics = collections.defaultdict(list)
    num_faulty_episodes = 0
    for episode in episodes:
        for k, v in episode.perf_stats.items():
            perf_stats[k].append(v)
        if episode.episode_faulty:
            num_faulty_episodes += 1
            continue
        episode_lengths.append(episode.episode_length)
        episode_rewards.append(episode.episode_reward)
        for k, v in episode.custom_metrics.items():
            custom_metrics[k].append(v)
        for (_, policy_id), reward in episode.agent_rewards.items():
            if policy_id != DEFAULT_POLICY_ID:
                policy_rewards[policy_id].append(reward)
        for k, v in episode.hist_data.items():
            hist_stats[k] += v
        for k, v in episode.media.items():
            episode_media[k].append(v)
        if hasattr(episode, 'connector_metrics'):
            for per_pipeline_metrics in episode.connector_metrics.values():
                for per_connector_metrics in per_pipeline_metrics.values():
                    for connector_metric_name, val in per_connector_metrics.items():
                        connector_metrics[connector_metric_name].append(val)
    if episode_rewards:
        min_reward = min(episode_rewards)
        max_reward = max(episode_rewards)
        avg_reward = np.mean(episode_rewards)
    else:
        min_reward = float('nan')
        max_reward = float('nan')
        avg_reward = float('nan')
    if episode_lengths:
        avg_length = np.mean(episode_lengths)
    else:
        avg_length = float('nan')
    hist_stats['episode_reward'] = episode_rewards
    hist_stats['episode_lengths'] = episode_lengths
    policy_reward_min = {}
    policy_reward_mean = {}
    policy_reward_max = {}
    for policy_id, rewards in policy_rewards.copy().items():
        policy_reward_min[policy_id] = np.min(rewards)
        policy_reward_mean[policy_id] = np.mean(rewards)
        policy_reward_max[policy_id] = np.max(rewards)
        hist_stats['policy_{}_reward'.format(policy_id)] = rewards
    for k, v_list in custom_metrics.copy().items():
        filt = [v for v in v_list if not np.any(np.isnan(v))]
        if keep_custom_metrics:
            custom_metrics[k] = filt
        else:
            custom_metrics[k + '_mean'] = np.mean(filt)
            if filt:
                custom_metrics[k + '_min'] = np.min(filt)
                custom_metrics[k + '_max'] = np.max(filt)
            else:
                custom_metrics[k + '_min'] = float('nan')
                custom_metrics[k + '_max'] = float('nan')
            del custom_metrics[k]
    for k, v_list in perf_stats.copy().items():
        perf_stats[k] = np.mean(v_list)
    mean_connector_metrics = dict()
    for k, v_list in connector_metrics.items():
        mean_connector_metrics[k] = np.mean(v_list)
    return dict(episode_reward_max=max_reward, episode_reward_min=min_reward, episode_reward_mean=avg_reward, episode_len_mean=avg_length, episode_media=dict(episode_media), episodes_this_iter=len(new_episodes), policy_reward_min=policy_reward_min, policy_reward_max=policy_reward_max, policy_reward_mean=policy_reward_mean, custom_metrics=dict(custom_metrics), hist_stats=dict(hist_stats), sampler_perf=dict(perf_stats), num_faulty_episodes=num_faulty_episodes, connector_metrics=mean_connector_metrics)