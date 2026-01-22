import warnings
from Bio import BiopythonDeprecationWarning
def find_states(markov_model, output):
    """Find states in the given Markov model output.

    Returns a list of (states, score) tuples.
    """
    mm = markov_model
    N = len(mm.states)
    lp_initial = np.log(mm.p_initial + VERY_SMALL_NUMBER)
    lp_transition = np.log(mm.p_transition + VERY_SMALL_NUMBER)
    lp_emission = np.log(mm.p_emission + VERY_SMALL_NUMBER)
    indexes = itemindex(mm.alphabet)
    output = [indexes[x] for x in output]
    results = _viterbi(N, lp_initial, lp_transition, lp_emission, output)
    for i in range(len(results)):
        states, score = results[i]
        results[i] = ([mm.states[x] for x in states], np.exp(score))
    return results