import tempfile
import os
import pytest
from ase.calculators.aims import Aims
from ase import Atoms
@pytest.mark.skip('legacy test with hardcoded paths and commands')
def test_aims_interface():
    aims_command = 'aims.x'
    aims_command_alternative = 'mpirun -np 4 fhiaims.x'
    outfilename = 'alternative_aims.out'
    outfilename_default = 'aims.out'
    command = '{0:s} > {1:s}'.format(aims_command, outfilename)
    command_default = '{0:s} > {1:s}'.format(aims_command, outfilename_default)
    legacy_command = 'aims.version.serial.x > aims.out'
    legacy_aims_command = legacy_command.split('>')[0].strip()
    legacy_outfilename = legacy_command.split('>')[-1].strip()
    calc = Aims()
    assert calc.command == legacy_command
    assert calc.outfilename == legacy_outfilename
    assert calc.aims_command == legacy_aims_command
    os.environ['ASE_AIMS_COMMAND'] = aims_command_alternative
    calc = Aims()
    assert calc.command == '{0} > {1}'.format(aims_command_alternative, outfilename_default)
    assert calc.outfilename == outfilename_default
    assert calc.aims_command == aims_command_alternative
    calc = Aims(run_command=command)
    assert calc.command == command
    assert calc.outfilename == outfilename
    assert calc.aims_command == aims_command
    calc = Aims(run_command=aims_command)
    assert calc.command == command_default
    assert calc.aims_command == aims_command
    assert calc.outfilename == outfilename_default
    calc = Aims(command=command)
    assert calc.command == command
    assert calc.outfilename == outfilename
    assert calc.aims_command == aims_command
    calc = Aims(aims_command=aims_command)
    assert calc.command == command_default
    assert calc.outfilename == outfilename_default
    assert calc.aims_command == aims_command
    calc = Aims(aims_command=aims_command, outfilename=outfilename)
    assert calc.command == command
    assert calc.outfilename == outfilename
    assert calc.aims_command == aims_command
    calc.command = command_default
    assert calc.outfilename == outfilename_default
    assert calc.aims_command == aims_command
    assert calc.command == command_default
    calc.aims_command = aims_command_alternative
    assert calc.aims_command == aims_command_alternative
    assert calc.outfilename == outfilename_default
    assert calc.command == '{} > {}'.format(aims_command_alternative, outfilename_default)
    calc.outfilename = outfilename
    assert calc.command == '{} > {}'.format(aims_command_alternative, outfilename)
    assert calc.aims_command == aims_command_alternative
    assert calc.outfilename == outfilename
    tmp_dir = tempfile.mkdtemp()
    water = Atoms('HOH', [(1, 0, 0), (0, 0, 0), (0, 1, 0)])
    calc = Aims(xc='PBE', output=['dipole'], sc_accuracy_etot=1e-06, sc_accuracy_eev=0.001, sc_accuracy_rho=1e-06, species_dir='/data/rittmeyer/FHIaims/species_defaults/light/', sc_accuracy_forces=0.0001, label=tmp_dir)
    try:
        calc.prepare_input_files()
        raise AssertionError
    except ValueError:
        pass
    calc.atoms = water
    calc.prepare_input_files()
    for f in ['control.in', 'geometry.in']:
        assert os.path.isfile(os.path.join(tmp_dir, f))