from ase.atoms import Atoms
def get_dbh24_initial_states(name):
    """ Returns initial DBH24 states
    """
    assert name in dbh24_reaction_list
    d = dbh24_reaction_list[name]
    initial = d['initial']
    return initial