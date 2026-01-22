from ase.atoms import Atoms
def get_dbh24_final_states(name):
    """ Returns final DBH24 states
    """
    assert name in dbh24_reaction_list
    d = dbh24_reaction_list[name]
    final = d['final']
    return final