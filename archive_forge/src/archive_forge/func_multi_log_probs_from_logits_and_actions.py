from ray.rllib.algorithms.impala.vtrace_tf import VTraceFromLogitsReturns, VTraceReturns
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
def multi_log_probs_from_logits_and_actions(policy_logits, actions, dist_class, model):
    """Computes action log-probs from policy logits and actions.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    ACTION_SPACE refers to the list of numbers each representing a number of
    actions.

    Args:
        policy_logits: A list with length of ACTION_SPACE of float32
            tensors of shapes [T, B, ACTION_SPACE[0]], ...,
            [T, B, ACTION_SPACE[-1]] with un-normalized log-probabilities
            parameterizing a softmax policy.
        actions: A list with length of ACTION_SPACE of tensors of shapes
            [T, B, ...], ..., [T, B, ...]
            with actions.
        dist_class: Python class of the action distribution.

    Returns:
        A list with length of ACTION_SPACE of float32 tensors of shapes
            [T, B], ..., [T, B] corresponding to the sampling log probability
            of the chosen action w.r.t. the policy.
    """
    log_probs = []
    for i in range(len(policy_logits)):
        p_shape = policy_logits[i].shape
        a_shape = actions[i].shape
        policy_logits_flat = torch.reshape(policy_logits[i], (-1,) + tuple(p_shape[2:]))
        actions_flat = torch.reshape(actions[i], (-1,) + tuple(a_shape[2:]))
        log_probs.append(torch.reshape(dist_class(policy_logits_flat, model).logp(actions_flat), a_shape[:2]))
    return log_probs