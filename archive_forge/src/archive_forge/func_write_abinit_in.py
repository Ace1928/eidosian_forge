import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def write_abinit_in(fd, atoms, param=None, species=None, pseudos=None):
    import copy
    from ase.calculators.calculator import kpts2mp
    from ase.calculators.abinit import Abinit
    if param is None:
        param = {}
    _param = copy.deepcopy(Abinit.default_parameters)
    _param.update(param)
    param = _param
    if species is None:
        species = sorted(set(atoms.numbers))
    inp = {}
    inp.update(param)
    for key in ['xc', 'smearing', 'kpts', 'pps', 'raw']:
        del inp[key]
    smearing = param.get('smearing')
    if 'tsmear' in param or 'occopt' in param:
        assert smearing is None
    if smearing is not None:
        inp['occopt'] = {'fermi-dirac': 3, 'gaussian': 7}[smearing[0].lower()]
        inp['tsmear'] = smearing[1]
    inp['natom'] = len(atoms)
    if 'nbands' in param:
        inp['nband'] = param['nbands']
        del inp['nbands']
    if param.get('pps') not in ['pawxml']:
        if 'ixc' not in param:
            inp['ixc'] = {'LDA': 7, 'PBE': 11, 'revPBE': 14, 'RPBE': 15, 'WC': 23}[param['xc']]
    magmoms = atoms.get_initial_magnetic_moments()
    if magmoms.any():
        inp['nsppol'] = 2
        fd.write('spinat\n')
        for n, M in enumerate(magmoms):
            fd.write('%.14f %.14f %.14f\n' % (0, 0, M))
    else:
        inp['nsppol'] = 1
    if param['kpts'] is not None:
        mp = kpts2mp(atoms, param['kpts'])
        fd.write('kptopt 1\n')
        fd.write('ngkpt %d %d %d\n' % tuple(mp))
        fd.write('nshiftk 1\n')
        fd.write('shiftk\n')
        fd.write('%.1f %.1f %.1f\n' % tuple((np.array(mp) + 1) % 2 * 0.5))
    valid_lists = (list, np.ndarray)
    for key in sorted(inp):
        value = inp[key]
        unit = keys_with_units.get(key)
        if unit is not None:
            if 'fs**2' in unit:
                value /= fs ** 2
            elif 'fs' in unit:
                value /= fs
        if isinstance(value, valid_lists):
            if isinstance(value[0], valid_lists):
                fd.write('{}\n'.format(key))
                for dim in value:
                    write_list(fd, dim, unit)
            else:
                fd.write('{}\n'.format(key))
                write_list(fd, value, unit)
        elif unit is None:
            fd.write('{} {}\n'.format(key, value))
        else:
            fd.write('{} {} {}\n'.format(key, value, unit))
    if param['raw'] is not None:
        if isinstance(param['raw'], str):
            raise TypeError('The raw parameter is a single string; expected a sequence of lines')
        for line in param['raw']:
            if isinstance(line, tuple):
                fd.write(' '.join(['%s' % x for x in line]) + '\n')
            else:
                fd.write('%s\n' % line)
    fd.write('#Definition of the unit cell\n')
    fd.write('acell\n')
    fd.write('%.14f %.14f %.14f Angstrom\n' % (1.0, 1.0, 1.0))
    fd.write('rprim\n')
    if atoms.cell.rank != 3:
        raise RuntimeError('Abinit requires a 3D cell, but cell is {}'.format(atoms.cell))
    for v in atoms.cell:
        fd.write('%.14f %.14f %.14f\n' % tuple(v))
    fd.write('chkprim 0 # Allow non-primitive cells\n')
    fd.write('#Definition of the atom types\n')
    fd.write('ntypat %d\n' % len(species))
    fd.write('znucl {}\n'.format(' '.join((str(Z) for Z in species))))
    fd.write('#Enumerate different atomic species\n')
    fd.write('typat')
    fd.write('\n')
    types = []
    for Z in atoms.numbers:
        for n, Zs in enumerate(species):
            if Z == Zs:
                types.append(n + 1)
    n_entries_int = 20
    for n, type in enumerate(types):
        fd.write(' %d' % type)
        if n > 1 and n % n_entries_int == 1:
            fd.write('\n')
    fd.write('\n')
    if pseudos is not None:
        listing = ',\n'.join(pseudos)
        line = f'pseudos "{listing}"\n'
        fd.write(line)
    fd.write('#Definition of the atoms\n')
    fd.write('xcart\n')
    for pos in atoms.positions / Bohr:
        fd.write('%.14f %.14f %.14f\n' % tuple(pos))
    fd.write('chkexit 1 # abinit.exit file in the running directory terminates after the current SCF\n')