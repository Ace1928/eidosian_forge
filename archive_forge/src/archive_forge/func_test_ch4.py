import pytest
from ase.build import molecule
from ase.calculators.calculator import get_calculator_class
from ase.units import Ry
from ase.utils import workdir
@pytest.mark.parametrize('spec', [inputs('gamess_us', label='ch4'), inputs('gaussian', xc='lda', basis='3-21G')], ids=lambda spec: spec.name)
def test_ch4(tmp_path, spec):
    with workdir(str(tmp_path), mkdir=True):
        e_ch4 = _calculate(spec, 'CH4')
        e_c2h2 = _calculate(spec, 'C2H2')
        e_h2 = _calculate(spec, 'H2')
        energy = e_ch4 - 0.5 * e_c2h2 - 1.5 * e_h2
        print(energy)
        ref_energy = -2.8
        assert abs(energy - ref_energy) < 0.3