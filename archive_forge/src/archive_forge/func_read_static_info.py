import re
import numpy as np
from ase.units import Bohr, Angstrom, Hartree, eV, Debye
def read_static_info(fd):
    results = {}

    def get_energy_unit(line):
        return {'[eV]': eV, '[H]': Hartree}[line.split()[1].rstrip(':')]
    for line in fd:
        if line.strip('*').strip().startswith('Brillouin zone'):
            results.update(read_static_info_kpoints(fd))
        elif line.startswith('Eigenvalues ['):
            unit = get_energy_unit(line)
            results.update(read_static_info_eigenvalues(fd, unit))
        elif line.startswith('Energy ['):
            unit = get_energy_unit(line)
            results.update(read_static_info_energy(fd, unit))
        elif line.startswith('Stress tensor'):
            assert line.split()[-1] == '[H/b^3]'
            stress = read_static_info_stress(fd)
            stress *= Hartree / Bohr ** 3
            results.update(stress=stress)
        elif line.startswith('Total Magnetic Moment'):
            if 0:
                line = next(fd)
                values = line.split()
                results['magmom'] = float(values[-1])
                line = next(fd)
                assert line.startswith('Local Magnetic Moments')
                line = next(fd)
                assert line.split() == ['Ion', 'mz']
                mag_moment = []
                for line in fd:
                    if line == '\n':
                        break
                    line = line.replace('\n', ' ')
                    values = line.split()
                    mag_moment.append(float(values[-1]))
                results['magmoms'] = np.array(mag_moment)
        elif line.startswith('Dipole'):
            assert line.split()[-1] == '[Debye]'
            dipole = [float(next(fd).split()[-1]) for i in range(3)]
            results['dipole'] = np.array(dipole) * Debye
        elif line.startswith('Forces'):
            forceunitspec = line.split()[-1]
            forceunit = {'[eV/A]': eV / Angstrom, '[H/b]': Hartree / Bohr}[forceunitspec]
            forces = []
            line = next(fd)
            assert line.strip().startswith('Ion')
            for line in fd:
                if line.strip().startswith('---'):
                    break
                tokens = line.split()[-3:]
                forces.append([float(f) for f in tokens])
            results['forces'] = np.array(forces) * forceunit
        elif line.startswith('Fermi'):
            tokens = line.split()
            unit = {'eV': eV, 'H': Hartree}[tokens[-1]]
            eFermi = float(tokens[-2]) * unit
            results['efermi'] = eFermi
    if 'ibz_k_points' not in results:
        results['ibz_k_points'] = np.zeros((1, 3))
        results['k_point_weights'] = np.ones(1)
    if 0:
        all_energies = results['eigenvalues'].ravel()
        all_occupations = results['occupations'].ravel()
        args = np.argsort(all_energies)
        for arg in args[::-1]:
            if all_occupations[arg] > 0.1:
                break
        eFermi = all_energies[arg]
        results['efermi'] = eFermi
    return results