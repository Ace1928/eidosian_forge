import pytest
@pytest.mark.skip('CLI must support calculator profiles')
def test_abinit_cmdline(abinit_factory, cli):
    cli.shell('\n    ase build -x fcc -a 4.04 Al |\n    ase -T run abinit -p xc=PBE,kpts=3.0,ecut=340,toldfe=1e-5,chksymbreak=0', 'abinit')