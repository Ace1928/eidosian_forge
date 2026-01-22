import numpy as np
def minimize_rotation_and_translation(target, atoms):
    """Minimize RMSD between atoms and target.
    
    Rotate and translate atoms to best match target.  For more details, see::
        
        Melander et al. J. Chem. Theory Comput., 2015, 11,1055
    """
    p = atoms.get_positions()
    p0 = target.get_positions()
    c = np.mean(p, axis=0)
    p -= c
    c0 = np.mean(p0, axis=0)
    p0 -= c0
    R = rotation_matrix_from_points(p.T, p0.T)
    atoms.set_positions(np.dot(p, R.T) + c0)