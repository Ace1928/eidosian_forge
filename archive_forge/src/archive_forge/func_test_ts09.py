from ase import io
from ase.calculators.vdwcorrection import vdWTkatchenko09prl
from ase.calculators.emt import EMT
from ase.build import bulk
def test_ts09(testdir):

    class FakeHirshfeldPartitioning:

        def __init__(self, calculator):
            self.calculator = calculator

        def initialize(self):
            pass

        def get_effective_volume_ratios(self):
            return [1]

        def get_calculator(self):
            return self.calculator

    class FakeDFTcalculator(EMT):

        def get_xc_functional(self):
            return 'PBE'
    a = 4.05
    al = bulk('Al', 'fcc', a=a)
    cc = FakeDFTcalculator()
    hp = FakeHirshfeldPartitioning(cc)
    c = vdWTkatchenko09prl(hp, [3])
    al.calc = c
    al.get_potential_energy()
    fname = 'out.traj'
    al.write(fname)
    io.read(fname)
    p = io.read(fname).calc.parameters
    p['calculator']
    p['xc']
    p['uncorrected_energy']