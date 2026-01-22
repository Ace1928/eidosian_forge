from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from .util import pad_center, fill_off_diagonal, is_positive_int, tiny, expand_to
from .util.exceptions import ParameterError
from .filters import get_window
from typing import Any, Iterable, List, Optional, Tuple, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _IntLike_co
def viterbi_discriminative(prob: np.ndarray, transition: np.ndarray, *, p_state: Optional[np.ndarray]=None, p_init: Optional[np.ndarray]=None, return_logp: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Viterbi decoding from discriminative state predictions.

    Given a sequence of conditional state predictions ``prob[s, t]``,
    indicating the conditional likelihood of state ``s`` given the
    observation at time ``t``, and a transition matrix ``transition[i, j]``
    which encodes the conditional probability of moving from state ``i``
    to state ``j``, the Viterbi algorithm computes the most likely sequence
    of states from the observations.

    This implementation uses the standard Viterbi decoding algorithm
    for observation likelihood sequences, under the assumption that
    ``P[Obs(t) | State(t) = s]`` is proportional to
    ``P[State(t) = s | Obs(t)] / P[State(t) = s]``, where the denominator
    is the marginal probability of state ``s`` occurring as given by ``p_state``.

    Note that because the denominator ``P[State(t) = s]`` is not explicitly
    calculated, the resulting probabilities (or log-probabilities) are not
    normalized.  If using the `return_logp=True` option (see below),
    be aware that the "probabilities" may not sum to (and may exceed) 1.

    Parameters
    ----------
    prob : np.ndarray [shape=(..., n_states, n_steps), non-negative]
        ``prob[s, t]`` is the probability of state ``s`` conditional on
        the observation at time ``t``.
        Must be non-negative and sum to 1 along each column.
    transition : np.ndarray [shape=(n_states, n_states), non-negative]
        ``transition[i, j]`` is the probability of a transition from i->j.
        Each row must sum to 1.
    p_state : np.ndarray [shape=(n_states,)]
        Optional: marginal probability distribution over states,
        must be non-negative and sum to 1.
        If not provided, a uniform distribution is assumed.
    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, it is assumed to be uniform.
    return_logp : bool
        If ``True``, return the log-likelihood of the state sequence.

    Returns
    -------
    Either ``states`` or ``(states, logp)``:
    states : np.ndarray [shape=(..., n_steps,)]
        The most likely state sequence.
        If ``prob`` contains multiple input channels,
        then each channel is decoded independently.
    logp : scalar [float] or np.ndarray
        If ``return_logp=True``, the (unnormalized) log probability
        of ``states`` given the observations.

    See Also
    --------
    viterbi :
        Viterbi decoding from observation likelihoods
    viterbi_binary :
        Viterbi decoding for multi-label, conditional state likelihoods

    Examples
    --------
    This example constructs a simple, template-based discriminative chord estimator,
    using CENS chroma as input features.

    .. note:: this chord model is not accurate enough to use in practice. It is only
            intended to demonstrate how to use discriminative Viterbi decoding.

    >>> # Create templates for major, minor, and no-chord qualities
    >>> maj_template = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0])
    >>> min_template = np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0])
    >>> N_template   = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1.]) / 4.
    >>> # Generate the weighting matrix that maps chroma to labels
    >>> weights = np.zeros((25, 12), dtype=float)
    >>> labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj',
    ...           'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
    ...           'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min',
    ...           'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
    ...           'N']
    >>> for c in range(12):
    ...     weights[c, :] = np.roll(maj_template, c) # c:maj
    ...     weights[c + 12, :] = np.roll(min_template, c)  # c:min
    >>> weights[-1] = N_template  # the last row is the no-chord class
    >>> # Make a self-loop transition matrix over 25 states
    >>> trans = librosa.sequence.transition_loop(25, 0.9)

    >>> # Load in audio and make features
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
    >>> # Suppress percussive elements
    >>> y = librosa.effects.harmonic(y, margin=4)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> # Map chroma (observations) to class (state) likelihoods
    >>> probs = np.exp(weights.dot(chroma))  # P[class | chroma] ~= exp(template' chroma)
    >>> probs /= probs.sum(axis=0, keepdims=True)  # probabilities must sum to 1 in each column
    >>> # Compute independent frame-wise estimates
    >>> chords_ind = np.argmax(probs, axis=0)
    >>> # And viterbi estimates
    >>> chords_vit = librosa.sequence.viterbi_discriminative(probs, trans)

    >>> # Plot the features and prediction map
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2)
    >>> librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', ax=ax[0])
    >>> librosa.display.specshow(weights, x_axis='chroma', ax=ax[1])
    >>> ax[1].set(yticks=np.arange(25) + 0.5, yticklabels=labels, ylabel='Chord')

    >>> # And plot the results
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(probs, x_axis='time', cmap='gray', ax=ax)
    >>> times = librosa.times_like(chords_vit)
    >>> ax.scatter(times, chords_ind + 0.25, color='lime', alpha=0.5, marker='+',
    ...            s=15, label='Independent')
    >>> ax.scatter(times, chords_vit - 0.25, color='deeppink', alpha=0.5, marker='o',
    ...            s=15, label='Viterbi')
    >>> ax.set(yticks=np.unique(chords_vit),
    ...        yticklabels=[labels[i] for i in np.unique(chords_vit)])
    >>> ax.legend()
    """
    n_states, n_steps = prob.shape[-2:]
    if transition.shape != (n_states, n_states):
        raise ParameterError(f'transition.shape={transition.shape}, must be (n_states, n_states)={(n_states, n_states)}')
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ParameterError('Invalid transition matrix: must be non-negative and sum to 1 on each row.')
    if np.any(prob < 0) or not np.allclose(prob.sum(axis=-2), 1):
        raise ParameterError('Invalid probability values: each column must sum to 1 and be non-negative')
    epsilon = tiny(prob)
    if p_state is None:
        p_state = np.empty(n_states)
        p_state.fill(1.0 / n_states)
    elif p_state.shape != (n_states,):
        raise ParameterError(f'Marginal distribution p_state must have shape (n_states,). Got p_state.shape={p_state.shape}')
    elif np.any(p_state < 0) or not np.allclose(p_state.sum(axis=-1), 1):
        raise ParameterError(f'Invalid marginal state distribution: p_state={p_state}')
    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1.0 / n_states)
    elif np.any(p_init < 0) or not np.allclose(p_init.sum(), 1) or p_init.shape != (n_states,):
        raise ParameterError(f'Invalid initial state distribution: p_init={p_init}')
    log_p_init = np.log(p_init + epsilon)
    log_trans = np.log(transition + epsilon)
    log_marginal = np.log(p_state + epsilon)
    log_marginal = expand_to(log_marginal, ndim=prob.ndim, axes=-2)
    log_prob = np.log(prob + epsilon) - log_marginal

    def _helper(lp):
        _state, logp = _viterbi(lp.T, log_trans, log_p_init)
        return (_state.T, logp)
    states: np.ndarray
    logp: np.ndarray
    if log_prob.ndim == 2:
        states, logp = _helper(log_prob)
    else:
        __viterbi = np.vectorize(_helper, otypes=[np.uint16, np.float64], signature='(s,t)->(t),(1)')
        states, logp = __viterbi(log_prob)
    logp = logp[..., 0]
    if return_logp:
        return (states, logp)
    return states