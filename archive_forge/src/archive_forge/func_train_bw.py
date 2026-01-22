import warnings
from Bio import BiopythonDeprecationWarning
def train_bw(states, alphabet, training_data, pseudo_initial=None, pseudo_transition=None, pseudo_emission=None, update_fn=None):
    """Train a MarkovModel using the Baum-Welch algorithm.

    Train a MarkovModel using the Baum-Welch algorithm.  states is a list
    of strings that describe the names of each state.  alphabet is a
    list of objects that indicate the allowed outputs.  training_data
    is a list of observations.  Each observation is a list of objects
    from the alphabet.

    pseudo_initial, pseudo_transition, and pseudo_emission are
    optional parameters that you can use to assign pseudo-counts to
    different matrices.  They should be matrices of the appropriate
    size that contain numbers to add to each parameter matrix, before
    normalization.

    update_fn is an optional callback that takes parameters
    (iteration, log_likelihood).  It is called once per iteration.
    """
    N, M = (len(states), len(alphabet))
    if not training_data:
        raise ValueError('No training data given.')
    if pseudo_initial is not None:
        pseudo_initial = np.asarray(pseudo_initial)
        if pseudo_initial.shape != (N,):
            raise ValueError('pseudo_initial not shape len(states)')
    if pseudo_transition is not None:
        pseudo_transition = np.asarray(pseudo_transition)
        if pseudo_transition.shape != (N, N):
            raise ValueError('pseudo_transition not shape len(states) X len(states)')
    if pseudo_emission is not None:
        pseudo_emission = np.asarray(pseudo_emission)
        if pseudo_emission.shape != (N, M):
            raise ValueError('pseudo_emission not shape len(states) X len(alphabet)')
    training_outputs = []
    indexes = itemindex(alphabet)
    for outputs in training_data:
        training_outputs.append([indexes[x] for x in outputs])
    lengths = [len(x) for x in training_outputs]
    if min(lengths) == 0:
        raise ValueError('I got training data with outputs of length 0')
    x = _baum_welch(N, M, training_outputs, pseudo_initial=pseudo_initial, pseudo_transition=pseudo_transition, pseudo_emission=pseudo_emission, update_fn=update_fn)
    p_initial, p_transition, p_emission = x
    return MarkovModel(states, alphabet, p_initial, p_transition, p_emission)