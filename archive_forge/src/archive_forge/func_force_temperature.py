import numpy as np
from ase.parallel import world
from ase import units
from ase.md.md import process_temperature
def force_temperature(atoms, temperature, unit='K'):
    """ force (nucl.) temperature to have a precise value

    Parameters:
    atoms: ase.Atoms
        the structure
    temperature: float
        nuclear temperature to set
    unit: str
        'K' or 'eV' as unit for the temperature
    """
    if unit == 'K':
        E_temp = temperature * units.kB
    elif unit == 'eV':
        E_temp = temperature
    else:
        raise UnitError("'{}' is not supported, use 'K' or 'eV'.".format(unit))
    if temperature > eps_temp:
        E_kin0 = atoms.get_kinetic_energy() / len(atoms) / 1.5
        gamma = E_temp / E_kin0
    else:
        gamma = 0.0
    atoms.set_momenta(atoms.get_momenta() * np.sqrt(gamma))