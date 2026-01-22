from math import pi, sqrt
def create_units(codata_version):
    """
    Function that creates a dictionary containing all units previously hard
    coded in ase.units depending on a certain CODATA version. Note that
    returned dict has attribute access it can be used in place of the module
    or to update your local or global namespace.

    Parameters:

    codata_version: str
        The CODATA version to be used. Implemented are

        * '1986'
        * '1998'
        * '2002'
        * '2006'
        * '2010'
        * '2014'

    Returns:

    units: dict
        Dictionary that contains all formerly hard coded variables from
        ase.units as key-value pairs. The dict supports attribute access.

    Raises:

    NotImplementedError
        If the required CODATA version is not known.
    """
    try:
        u = Units(CODATA[codata_version])
    except KeyError:
        raise NotImplementedError('CODATA version "{0}" not implemented'.format(codata_version))
    u['_eps0'] = 1 / u['_mu0'] / u['_c'] ** 2
    u['_hbar'] = u['_hplanck'] / (2 * pi)
    u['Ang'] = u['Angstrom'] = 1.0
    u['nm'] = 10.0
    u['Bohr'] = 40000000000.0 * pi * u['_eps0'] * u['_hbar'] ** 2 / u['_me'] / u['_e'] ** 2
    u['eV'] = 1.0
    u['Hartree'] = u['_me'] * u['_e'] ** 3 / 16 / pi ** 2 / u['_eps0'] ** 2 / u['_hbar'] ** 2
    u['kJ'] = 1000.0 / u['_e']
    u['kcal'] = 4.184 * u['kJ']
    u['mol'] = u['_Nav']
    u['Rydberg'] = 0.5 * u['Hartree']
    u['Ry'] = u['Rydberg']
    u['Ha'] = u['Hartree']
    u['second'] = 10000000000.0 * sqrt(u['_e'] / u['_amu'])
    u['fs'] = 1e-15 * u['second']
    u['kB'] = u['_k'] / u['_e']
    u['Pascal'] = 1 / u['_e'] / 1e+30
    u['GPa'] = 1000000000.0 * u['Pascal']
    u['bar'] = 100000.0 * u['Pascal']
    u['Debye'] = 1.0 / 100000000000.0 / u['_e'] / u['_c']
    u['alpha'] = u['_e'] ** 2 / (4 * pi * u['_eps0']) / u['_hbar'] / u['_c']
    u['invcm'] = 100 * u['_c'] * u['_hplanck'] / u['_e']
    u['_aut'] = u['_hbar'] / (u['alpha'] ** 2 * u['_me'] * u['_c'] ** 2)
    u['_auv'] = u['_e'] ** 2 / u['_hbar'] / (4 * pi * u['_eps0'])
    u['_auf'] = u['alpha'] ** 3 * u['_me'] ** 2 * u['_c'] ** 3 / u['_hbar']
    u['_aup'] = u['alpha'] ** 5 * u['_me'] ** 4 * u['_c'] ** 5 / u['_hbar'] ** 3
    u['AUT'] = u['second'] * u['_aut']
    u['m'] = 10000000000.0 * u['Ang']
    u['kg'] = 1.0 / u['_amu']
    u['s'] = u['second']
    u['A'] = 1.0 / u['_e'] / u['s']
    u['J'] = u['kJ'] / 1000
    u['C'] = 1.0 / u['_e']
    return u