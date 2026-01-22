import re
import numpy as np
from ase.units import Bohr, Angstrom, Hartree, eV, Debye
def read_static_info_eigenvalues(fd, energy_unit):
    values_sknx = {}
    nbands = 0
    fermilevel = None
    for line in fd:
        line = line.strip()
        if line.startswith('#'):
            continue
        if not line[:1].isdigit():
            m = re.match('Fermi energy\\s*=\\s*(\\S+)', line)
            if m is not None:
                fermilevel = float(m.group(1)) * energy_unit
            break
        tokens = line.split()
        nbands = max(nbands, int(tokens[0]))
        energy = float(tokens[2]) * energy_unit
        occupation = float(tokens[3])
        values_sknx.setdefault(tokens[1], []).append((energy, occupation))
    nspins = len(values_sknx)
    if nspins == 1:
        val = [values_sknx['--']]
    else:
        val = [values_sknx['up'], values_sknx['dn']]
    val = np.array(val, float)
    nkpts, remainder = divmod(len(val[0]), nbands)
    assert remainder == 0
    eps_skn = val[:, :, 0].reshape(nspins, nkpts, nbands)
    occ_skn = val[:, :, 1].reshape(nspins, nkpts, nbands)
    eps_skn = eps_skn.transpose(1, 0, 2).copy()
    occ_skn = occ_skn.transpose(1, 0, 2).copy()
    assert eps_skn.flags.contiguous
    d = dict(nspins=nspins, nkpts=nkpts, nbands=nbands, eigenvalues=eps_skn, occupations=occ_skn)
    if fermilevel is not None:
        d.update(efermi=fermilevel)
    return d