import pytest
from ..config import ET_ROOT
from ..client import _etrequest, get_project, check_available_version
@pytest.mark.skipif(no_cxn, reason='No connection')
def test_check_available():
    repo = 'invalidrepo'
    res = check_available_version(repo, '0.1.0')
    assert res is None
    repo = 'github/hub'
    res = check_available_version(repo, '0.1.0')
    assert 'version' in res
    res = check_available_version(repo, res['version'])
    assert 'version' in res
    res = check_available_version(repo, '1000.1.0')
    assert 'version' in res