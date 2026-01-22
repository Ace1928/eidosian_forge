import warnings
from Bio import BiopythonDeprecationWarning
def train_visible(states, alphabet, training_data, pseudo_initial=None, pseudo_transition=None, pseudo_emission=None):
    """Train a visible MarkovModel using maximum likelihoood estimates for each of the parameters.

    Train a visible MarkovModel using maximum likelihoood estimates
    for each of the parameters.  states is a list of strings that
    describe the names of each state.  alphabet is a list of objects
    that indicate the allowed outputs.  training_data is a list of
    (outputs, observed states) where outputs is a list of the emission
    from the alphabet, and observed states is a list of states from
    states.

    pseudo_initial, pseudo_transition, and pseudo_emission are
    optional parameters that you can use to assign pseudo-counts to
    different matrices.  They should be matrices of the appropriate
    size that contain numbers to add to each parameter matrix.
    """
    N, M = (len(states), len(alphabet))
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
    training_states, training_outputs = ([], [])
    states_indexes = itemindex(states)
    outputs_indexes = itemindex(alphabet)
    for toutputs, tstates in training_data:
        if len(tstates) != len(toutputs):
            raise ValueError('states and outputs not aligned')
        training_states.append([states_indexes[x] for x in tstates])
        training_outputs.append([outputs_indexes[x] for x in toutputs])
    x = _mle(N, M, training_outputs, training_states, pseudo_initial, pseudo_transition, pseudo_emission)
    p_initial, p_transition, p_emission = x
    return MarkovModel(states, alphabet, p_initial, p_transition, p_emission)