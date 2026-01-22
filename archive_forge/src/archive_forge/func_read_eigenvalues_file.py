import re
import numpy as np
from ase.units import Bohr, Angstrom, Hartree, eV, Debye
def read_eigenvalues_file(fd):
    unit = None
    for line in fd:
        m = re.match('Eigenvalues\\s*\\[(.+?)\\]', line)
        if m is not None:
            unit = m.group(1)
            break
    line = next(fd)
    assert line.strip().startswith('#st'), line
    kpts = []
    eigs = []
    occs = []
    for line in fd:
        m = re.match('#k.*?\\(\\s*(.+?),\\s*(.+?),\\s*(.+?)\\)', line)
        if m:
            k = m.group(1, 2, 3)
            kpts.append(np.array(k, float))
            eigs.append({})
            occs.append({})
        else:
            m = re.match('\\s*\\d+\\s*(\\S+)\\s*(\\S+)\\s*(\\S+)', line)
            if m is None:
                m = re.match('Fermi energy\\s*=\\s*(\\S+)\\s*', line)
                assert m is not None
            else:
                spin, eig, occ = m.group(1, 2, 3)
                if not eigs:
                    eigs.append({})
                    occs.append({})
                eigs[-1].setdefault(spin, []).append(float(eig))
                occs[-1].setdefault(spin, []).append(float(occ))
    nkpts = len(kpts)
    nspins = len(eigs[0])
    nbands = len(eigs[0][spin])
    kptsarr = np.array(kpts, float)
    eigsarr = np.empty((nkpts, nspins, nbands))
    occsarr = np.empty((nkpts, nspins, nbands))
    arrs = [eigsarr, occsarr]
    for arr in arrs:
        arr.fill(np.nan)
    for k in range(nkpts):
        for arr, lst in [(eigsarr, eigs), (occsarr, occs)]:
            arr[k, :, :] = [lst[k][sp] for sp in (['--'] if nspins == 1 else ['up', 'dn'])]
    for arr in arrs:
        assert not np.isnan(arr).any()
    eigsarr *= {'H': Hartree, 'eV': eV}[unit]
    return (kptsarr, eigsarr, occsarr)