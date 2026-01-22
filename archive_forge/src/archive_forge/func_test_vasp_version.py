from ase.calculators.vasp import get_vasp_version
def test_vasp_version():
    assert get_vasp_version(vasp_sample_header) == '6.1.2'