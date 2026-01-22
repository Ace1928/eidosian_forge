from ase import Atoms
from ase.calculators.crystal import CRYSTAL
def test_graphene(testdir):
    with open('basis', 'w') as fd:
        fd.write('6 4\n    0 0 6 2.0 1.0\n     3048.0 0.001826\n     456.4 0.01406\n     103.7 0.06876\n     29.23 0.2304\n     9.349 0.4685\n     3.189 0.3628\n    0 1 2 4.0 1.0\n     3.665 -0.3959 0.2365\n     0.7705 1.216 0.8606\n    0 1 1 0.0 1.0\n     0.26 1.0 1.0\n    0 3 1 0.0 1.0\n     0.8 1.0\n    ')
    geom = Atoms('C2', cell=[[2.1680326, -1.2517142, 0.0], [0.0, 2.5034284, 0.0], [0.0, 0.0, 500.0]], positions=[(-0.722677550504, -1.251714234963, 0.0), (-1.445355101009, 0.0, 0.0)], pbc=[True, True, False])
    geom.calc = CRYSTAL(label='graphene', guess=True, xc='PBE', kpts=(1, 1, 1), otherkeys=['scfdir', 'anderson', ['maxcycles', '500'], ['toldee', '6'], ['tolinteg', '7 7 7 7 14'], ['fmixing', '95']])
    final_energy = geom.get_potential_energy()
    assert abs(final_energy + 2063.13266758) < 1.0