import collections
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
@writer
def write_elk_in(fd, atoms, parameters=None):
    if parameters is None:
        parameters = {}
    parameters = dict(parameters)
    species_path = parameters.pop('species_dir', None)
    if parameters.get('spinpol') is None:
        if atoms.get_initial_magnetic_moments().any():
            parameters['spinpol'] = True
    if 'xctype' in parameters:
        if 'xc' in parameters:
            raise RuntimeError("You can't use both 'xctype' and 'xc'!")
    if parameters.get('autokpt'):
        if 'kpts' in parameters:
            raise RuntimeError("You can't use both 'autokpt' and 'kpts'!")
        if 'ngridk' in parameters:
            raise RuntimeError("You can't use both 'autokpt' and 'ngridk'!")
    if 'ngridk' in parameters:
        if 'kpts' in parameters:
            raise RuntimeError("You can't use both 'ngridk' and 'kpts'!")
    if parameters.get('autoswidth'):
        if 'smearing' in parameters:
            raise RuntimeError("You can't use both 'autoswidth' and 'smearing'!")
        if 'swidth' in parameters:
            raise RuntimeError("You can't use both 'autoswidth' and 'swidth'!")
    inp = {}
    inp.update(parameters)
    if 'xc' in parameters:
        xctype = {'LDA': 3, 'PBE': 20, 'REVPBE': 21, 'PBESOL': 22, 'WC06': 26, 'AM05': 30, 'mBJLDA': (100, 208, 12)}[parameters['xc']]
        inp['xctype'] = xctype
        del inp['xc']
    if 'kpts' in parameters:
        from ase.calculators.calculator import kpts2mp
        mp = kpts2mp(atoms, parameters['kpts'])
        inp['ngridk'] = tuple(mp)
        vkloff = []
        for nk in mp:
            if nk % 2 == 0:
                vkloff.append(0.5)
            else:
                vkloff.append(0)
        inp['vkloff'] = vkloff
        del inp['kpts']
    if 'smearing' in parameters:
        name = parameters.smearing[0].lower()
        if name == 'methfessel-paxton':
            stype = parameters.smearing[2]
        else:
            stype = {'gaussian': 0, 'fermi-dirac': 3}[name]
        inp['stype'] = stype
        inp['swidth'] = parameters.smearing[1]
        del inp['smearing']
    for key, value in inp.items():
        if key in elk_parameters:
            inp[key] /= elk_parameters[key]
    for key, value in inp.items():
        fd.write('%s\n' % key)
        if isinstance(value, bool):
            fd.write('.%s.\n\n' % ('false', 'true')[value])
        elif isinstance(value, (int, float)):
            fd.write('%s\n\n' % value)
        else:
            fd.write('%s\n\n' % ' '.join([str(x) for x in value]))
    fd.write('avec\n')
    for vec in atoms.cell:
        fd.write('%.14f %.14f %.14f\n' % tuple(vec / Bohr))
    fd.write('\n')
    species = {}
    symbols = []
    for a, (symbol, m) in enumerate(zip(atoms.get_chemical_symbols(), atoms.get_initial_magnetic_moments())):
        if symbol in species:
            species[symbol].append((a, m))
        else:
            species[symbol] = [(a, m)]
            symbols.append(symbol)
    fd.write('atoms\n%d\n' % len(species))
    scaled = np.linalg.solve(atoms.cell.T, atoms.positions.T).T
    for symbol in symbols:
        fd.write("'%s.in' : spfname\n" % symbol)
        fd.write('%d\n' % len(species[symbol]))
        for a, m in species[symbol]:
            fd.write('%.14f %.14f %.14f 0.0 0.0 %.14f\n' % (tuple(scaled[a]) + (m,)))
    if species_path is not None:
        fd.write(f"sppath\n'{species_path}/'\n\n")