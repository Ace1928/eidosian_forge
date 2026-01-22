from nipype.interfaces.ants.base import Info
import pytest
@pytest.mark.parametrize('raw_info, version', ANTS_VERSIONS)
def test_version_parser(raw_info, version):
    assert Info.parse_version(raw_info) == version