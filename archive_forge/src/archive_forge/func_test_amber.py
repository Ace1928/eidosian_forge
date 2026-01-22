import subprocess
from ase import Atoms
from ase.calculators.amber import Amber
def test_amber(factories):
    """Test that amber calculator works.

    This is conditional on the existence of the $AMBERHOME/bin/sander
    executable.
    """
    factories.require('amber')
    with open('mm.in', 'w') as outfile:
        outfile.write('    zero step md to get energy and force\n    &cntrl\n    imin=0, nstlim=0,  ntx=1 !0 step md\n    cut=100, ntb=0,          !non-periodic\n    ntpr=1,ntwf=1,ntwe=1,ntwx=1 ! (output frequencies)\n    &end\n    END\n    ')
    with open('tleap.in', 'w') as outfile:
        outfile.write('    source leaprc.protein.ff14SB\n    source leaprc.gaff\n    source leaprc.water.tip3p\n    mol = loadpdb 2h2o.pdb\n    saveamberparm mol 2h2o.top h2o.inpcrd\n    quit\n    ')
    subprocess.call('tleap -f tleap.in'.split())
    atoms = Atoms('OH2OH2', [[-0.956, -0.121, 0], [-1.308, 0.77, 0], [0.0, 0.0, 0], [3.903, 0.0, 0], [4.215, -0.497, -0.759], [4.215, -0.497, 0.759]])
    calc = Amber(amber_exe='sander -O ', infile='mm.in', outfile='mm.out', topologyfile='2h2o.top', incoordfile='mm.crd')
    calc.write_coordinates(atoms, 'mm.crd')
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert abs(e + 0.046799672) < 0.005