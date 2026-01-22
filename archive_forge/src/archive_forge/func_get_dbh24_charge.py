from ase.atoms import Atoms
def get_dbh24_charge(name):
    """ Returns the total charge of DBH24 systems.
    """
    assert name in dbh24
    d = data[name]
    charge = d['charge']
    return charge