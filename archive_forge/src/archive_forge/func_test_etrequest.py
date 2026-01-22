import pytest
from ..config import ET_ROOT
from ..client import _etrequest, get_project, check_available_version
@pytest.mark.skipif(no_cxn, reason='No connection')
def test_etrequest():
    endpoint = 'http://fakeendpoint/'
    with pytest.raises(RuntimeError):
        _etrequest(endpoint, method='get')
    assert _etrequest(ET_ROOT)
    endpoint = 'https://google.com'
    with pytest.raises(RuntimeError):
        _etrequest(endpoint, timeout=0.001)
    assert _etrequest(endpoint)