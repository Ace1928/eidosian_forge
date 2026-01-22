def test_eos():
    import numpy as np
    import scipy
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.eos import EquationOfState as EOS, eos_names
    scipy
    b = bulk('Al', 'fcc', a=4.0, orthorhombic=True)
    b.calc = EMT()
    cell = b.get_cell()
    volumes = []
    energies = []
    for x in np.linspace(0.98, 1.01, 5):
        b.set_cell(cell * x, scale_atoms=True)
        volumes.append(b.get_volume())
        energies.append(b.get_potential_energy())
    results = []
    for name in eos_names:
        if name == 'antonschmidt':
            continue
        eos = EOS(volumes, energies, name)
        v, e, b = eos.fit()
        print('{0:20} {1:.8f} {2:.8f} {3:.8f} '.format(name, v, e, b))
        assert abs(v - 31.86587) < 0.0004
        assert abs(e - -0.00976187802) < 5e-07
        assert abs(b - 0.246812688) < 0.0002
        results.append((v, e, b))
    print(np.ptp(results, 0))
    print(np.mean(results, 0))