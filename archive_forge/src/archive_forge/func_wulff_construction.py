import numpy as np
def wulff_construction(symbol, surfaces, energies, size, structure, rounding='closest', latticeconstant=None, debug=False, maxiter=100):
    """Create a cluster using the Wulff construction.

    A cluster is created with approximately the number of atoms
    specified, following the Wulff construction, i.e. minimizing the
    surface energy of the cluster.

    Parameters:

    symbol: The chemical symbol (or atomic number) of the desired element.

    surfaces: A list of surfaces. Each surface is an (h, k, l) tuple or
    list of integers.

    energies: A list of surface energies for the surfaces.

    size: The desired number of atoms.

    structure: The desired crystal structure.  One of the strings
    "fcc", "bcc", or "sc".

    rounding (optional): Specifies what should be done if no Wulff
    construction corresponds to exactly the requested number of atoms.
    Should be a string, either "above", "below" or "closest" (the
    default), meaning that the nearest cluster above or below - or the
    closest one - is created instead.

    latticeconstant (optional): The lattice constant.  If not given,
    extracted from ase.data.

    debug (optional): If non-zero, information about the iteration towards
    the right cluster size is printed.
    """
    if debug:
        print('Wulff: Aiming for cluster with %i atoms (%s)' % (size, rounding))
        if rounding not in ['above', 'below', 'closest']:
            raise ValueError('Invalid rounding: %s' % rounding)
    if isinstance(structure, str):
        if structure == 'fcc':
            from ase.cluster.cubic import FaceCenteredCubic as structure
        elif structure == 'bcc':
            from ase.cluster.cubic import BodyCenteredCubic as structure
        elif structure == 'sc':
            from ase.cluster.cubic import SimpleCubic as structure
        elif structure == 'hcp':
            from ase.cluster.hexagonal import HexagonalClosedPacked as structure
        elif structure == 'graphite':
            from ase.cluster.hexagonal import Graphite as structure
        else:
            error = 'Crystal structure %s is not supported.' % structure
            raise NotImplementedError(error)
    nsurf = len(surfaces)
    if len(energies) != nsurf:
        raise ValueError('The energies array should contain %d values.' % (nsurf,))
    energies = np.array(energies)
    atoms = structure(symbol, surfaces, 5 * np.ones(len(surfaces), int), latticeconstant=latticeconstant)
    for i, s in enumerate(surfaces):
        d = atoms.get_layer_distance(s)
        energies[i] /= d
    wanted_size = size ** (1.0 / 3.0)
    max_e = max(energies)
    factor = wanted_size / max_e
    atoms, layers = make_atoms(symbol, surfaces, energies, factor, structure, latticeconstant)
    if len(atoms) == 0:
        if debug:
            print('First try made an empty cluster, trying again.')
        factor = 1 / energies.min()
        atoms, layers = make_atoms(symbol, surfaces, energies, factor, structure, latticeconstant)
        if len(atoms) == 0:
            raise RuntimeError('Failed to create a finite cluster.')
    old_factor = factor
    old_layers = layers
    old_atoms = atoms
    factor *= (size / len(atoms)) ** (1.0 / 3.0)
    atoms, layers = make_atoms(symbol, surfaces, energies, factor, structure, latticeconstant)
    if len(atoms) == 0:
        print('Second guess gave an empty cluster, discarding it.')
        atoms = old_atoms
        factor = old_factor
        layers = old_layers
    else:
        del old_atoms
    below = above = None
    if len(atoms) <= size:
        below = atoms
    if len(atoms) >= size:
        above = atoms
    iter = 0
    while below is None or above is None:
        if len(atoms) < size:
            if debug:
                print('Making a larger cluster.')
            factor = ((layers + 0.5 + delta) / energies).min()
            atoms, new_layers = make_atoms(symbol, surfaces, energies, factor, structure, latticeconstant)
            assert (new_layers - layers).max() == 1
            assert (new_layers - layers).min() >= 0
            layers = new_layers
        else:
            if debug:
                print('Making a smaller cluster.')
            factor = ((layers - 0.5 - delta) / energies).max()
            atoms, new_layers = make_atoms(symbol, surfaces, energies, factor, structure, latticeconstant)
            assert (new_layers - layers).max() <= 0
            assert (new_layers - layers).min() == -1
            layers = new_layers
        if len(atoms) <= size:
            below = atoms
        if len(atoms) >= size:
            above = atoms
        iter += 1
        if iter == maxiter:
            raise RuntimeError('Runaway iteration.')
    if rounding == 'below':
        if debug:
            print('Choosing smaller cluster with %i atoms' % len(below))
        return below
    elif rounding == 'above':
        if debug:
            print('Choosing larger cluster with %i atoms' % len(above))
        return above
    else:
        assert rounding == 'closest'
        if len(above) - size < size - len(below):
            atoms = above
        else:
            atoms = below
        if debug:
            print('Choosing closest cluster with %i atoms' % len(atoms))
        return atoms