from __future__ import annotations
import numpy as np
def hellinger_distance(dist_p: dict, dist_q: dict) -> float:
    """Computes the Hellinger distance between
    two counts distributions.

    Parameters:
        dist_p (dict): First dict of counts.
        dist_q (dict): Second dict of counts.

    Returns:
        float: Distance

    References:
        `Hellinger Distance @ wikipedia <https://en.wikipedia.org/wiki/Hellinger_distance>`_
    """
    p_sum = sum(dist_p.values())
    q_sum = sum(dist_q.values())
    p_normed = {}
    for key, val in dist_p.items():
        p_normed[key] = val / p_sum
    q_normed = {}
    for key, val in dist_q.items():
        q_normed[key] = val / q_sum
    total = 0
    for key, val in p_normed.items():
        if key in q_normed:
            total += (np.sqrt(val) - np.sqrt(q_normed[key])) ** 2
            del q_normed[key]
        else:
            total += val
    total += sum(q_normed.values())
    dist = np.sqrt(total) / np.sqrt(2)
    return dist