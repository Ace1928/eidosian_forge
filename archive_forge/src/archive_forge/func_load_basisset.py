import itertools
def load_basisset(basis, element):
    """Extracts basis set data from the Basis Set Exchange library.

    Args:
        basis (str): name of the basis set
        element (str): atomic symbol of the chemical element

    Returns:
        dict[str, list]: dictionary containing orbital names, and the exponents and contraction
        coefficients of a basis function

    **Example**

    >>> basis = '6-31g'
    >>> element = 'He'
    >>> basis = qml.qchem.load_basisset(basis, element)
    >>> basis
    {'orbitals': ['S', 'S'],
     'exponents': [[38.421634, 5.77803, 1.241774], [0.297964]],
     'coefficients': [[0.04013973935, 0.261246097, 0.7931846246], [1.0]]}
    """
    try:
        import basis_set_exchange as bse
    except ImportError as Error:
        raise ImportError('This feature requires basis_set_exchange. It can be installed with: pip install basis-set-exchange.') from Error
    orbital_map = {'[0]': 'S', '[0, 1]': 'SP', '[1]': 'P', '[2]': 'D', '[3]': 'F', '[4]': 'G', '[5]': 'H'}
    element = str(atomic_numbers[element])
    data = bse.get_basis(basis)['elements'][element]['electron_shells']
    orbitals = []
    exponents = []
    coefficients = []
    for term in data:
        if orbital_map[str(term['angular_momentum'])] == 'SP':
            orbitals.append(['S', 'P'])
        else:
            orbitals.append([orbital_map[str(term['angular_momentum'])]] * len(term['coefficients']))
        exponents.append([list(map(float, item)) for item in [term['exponents']] * len(term['coefficients'])])
        coefficients.append([list(map(float, item)) for item in term['coefficients']])
    return {'orbitals': list(itertools.chain(*orbitals)), 'exponents': list(itertools.chain(*exponents)), 'coefficients': list(itertools.chain(*coefficients))}