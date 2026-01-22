from ase.atoms import Atoms
def get_s22x5_id(name):
    """Get main name and relative separation distance of an S22x5 system.
    """
    s22_name = name[:-4]
    dist = name[-3:]
    return (s22_name, dist)