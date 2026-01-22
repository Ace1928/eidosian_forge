from ase.atoms import Atoms
def get_number_of_dimer_atoms(name, dist=None):
    """Returns the number of atoms in each s22 dimer as a list; [x,y].
    """
    s22_, s22x5_, s22_name, dist_ = identify_s22_sys(name, dist)
    return data[s22_name]['dimer atoms']