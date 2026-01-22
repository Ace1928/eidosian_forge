import numpy as np
from numpy.random import RandomState
import pytest
from ase.phonons import Phonons
from ase.data import atomic_numbers
from ase.optimize import FIRE
from ase.build import bulk
from ase.md.velocitydistribution import PhononHarmonics
@pytest.mark.slow
def test_phonon_md_init(asap3, testdir):
    EMT = asap3.EMT
    rng = RandomState(17)
    atoms = bulk('Pd')
    atoms *= (3, 3, 3)
    avail = [atomic_numbers[sym] for sym in ['Ni', 'Cu', 'Pd', 'Ag', 'Pt', 'Au']]
    atoms.numbers[:] = rng.choice(avail, size=len(atoms))
    atoms.calc = EMT()
    with FIRE(atoms, trajectory='relax.traj') as opt:
        opt.run(fmax=0.001)
    positions0 = atoms.positions.copy()
    phonons = Phonons(atoms, EMT(), supercell=(1, 1, 1), delta=0.05)
    try:
        phonons.run()
        phonons.read()
    finally:
        phonons.clean()
    matrices = phonons.get_force_constant()
    K = matrices[0]
    T = 300
    atoms.calc = EMT()
    Epotref = atoms.get_potential_energy()
    temps = []
    Epots = []
    Ekins = []
    Etots = []
    for i in range(24):
        PhononHarmonics(atoms, K, temperature_K=T, quantum=True, rng=np.random.RandomState(888 + i))
        Epot = atoms.get_potential_energy() - Epotref
        Ekin = atoms.get_kinetic_energy()
        Ekins.append(Ekin)
        Epots.append(Epot)
        Etots.append(Ekin + Epot)
        temps.append(atoms.get_temperature())
        atoms.positions[:] = positions0
        print('energies', Epot, Ekin, Epot + Ekin)
    Epotmean = np.mean(Epots)
    Ekinmean = np.mean(Ekins)
    Tmean = np.mean(temps)
    Terr = abs(Tmean - T)
    relative_imbalance = abs(Epotmean - Ekinmean) / (Epotmean + Ekinmean)
    print('epotmean', Epotmean)
    print('ekinmean', Ekinmean)
    print('rel imbalance', relative_imbalance)
    print('Tmean', Tmean, 'Tref', T, 'err', Terr)
    assert Terr < 0.1 * T, Terr
    assert relative_imbalance < 0.1, relative_imbalance
    if 0:
        import matplotlib.pyplot as plt
        I = np.arange(len(Epots))
        plt.plot(I, Epots, 'o', label='pot')
        plt.plot(I, Ekins, 'o', label='kin')
        plt.plot(I, Etots, 'o', label='tot')
        plt.show()