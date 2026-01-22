from ase.atoms import Atoms
def get_dbh24_magmoms(name):
    """Returns the magnetic moments of DBH24 systems.
    """
    if name not in data:
        raise KeyError('System %s not in database.' % name)
    else:
        return data[name]['magmoms']