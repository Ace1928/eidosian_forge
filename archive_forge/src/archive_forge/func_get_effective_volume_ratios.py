from ase import io
from ase.calculators.vdwcorrection import vdWTkatchenko09prl
from ase.calculators.emt import EMT
from ase.build import bulk
def get_effective_volume_ratios(self):
    return [1]