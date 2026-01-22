import pytest
from ase.calculators.gromacs import parse_gromacs_version, get_gromacs_version
@pytest.mark.calculator_lite
@pytest.mark.calculator('gromacs')
def test_gromacs(factory):
    GRO_INIT_FILE = 'hise_box.gro'
    with open(GRO_INIT_FILE, 'w') as outfile:
        outfile.write(data)
    calc = factory.calc(force_field='charmm27', define='-DFLEXIBLE', integrator='cg', nsteps='10000', nstfout='10', nstlog='10', nstenergy='10', nstlist='10', ns_type='grid', pbc='xyz', rlist='0.7', coulombtype='PME-Switch', rcoulomb='0.6', vdwtype='shift', rvdw='0.6', rvdw_switch='0.55', DispCorr='Ener')
    calc.set_own_params_runs('init_structure', GRO_INIT_FILE)
    calc.generate_topology_and_g96file()
    calc.write_input()
    calc.generate_gromacs_run_file()
    calc.run()
    atoms = calc.get_atoms()
    final_energy = calc.get_potential_energy(atoms)
    final_energy_ref = -4.175
    tolerance = 0.01
    assert abs(final_energy - final_energy_ref) < tolerance