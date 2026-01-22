from ase.atoms import Atoms
def get_dbh24_Vb(name):
    """ Returns backward DBH24 TST barrier in kcal/mol
    """
    assert name in dbh24
    d = data[name]
    Vb = d['Vb']
    return Vb