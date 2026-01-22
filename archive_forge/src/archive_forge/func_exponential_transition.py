from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
def exponential_transition(from_intervals, to_intervals, transition_lambda, threshold=np.spacing(1), norm=True):
    """
    Exponential tempo transition.

    Parameters
    ----------
    from_intervals : numpy array
        Intervals where the transitions originate from.
    to_intervals :  : numpy array
        Intervals where the transitions terminate.
    transition_lambda : float
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat/bar to the next one). If None,
        allow only transitions from/to the same interval.
    threshold : float, optional
        Set transition probabilities below this threshold to zero.
    norm : bool, optional
        Normalize the emission probabilities to sum 1.

    Returns
    -------
    probabilities : numpy array, shape (num_from_intervals, num_to_intervals)
        Probability of each transition from an interval to another.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """
    if transition_lambda is None:
        return np.diag(np.diag(np.ones((len(from_intervals), len(to_intervals)))))
    ratio = to_intervals.astype(np.float) / from_intervals.astype(np.float)[:, np.newaxis]
    prob = np.exp(-transition_lambda * abs(ratio - 1.0))
    prob[prob <= threshold] = 0
    if norm:
        prob /= np.sum(prob, axis=1)[:, np.newaxis]
    return prob