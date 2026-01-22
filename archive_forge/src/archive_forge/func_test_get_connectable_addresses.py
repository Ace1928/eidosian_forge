import pytest
from testpath import modified_env
from jeepney import bus
def test_get_connectable_addresses():
    a = list(bus.get_connectable_addresses('unix:path=/run/user/1000/bus'))
    assert a == ['/run/user/1000/bus']
    a = list(bus.get_connectable_addresses('unix:abstract=/tmp/foo'))
    assert a == ['\x00/tmp/foo']
    with pytest.raises(RuntimeError):
        list(bus.get_connectable_addresses('unix:tmpdir=/tmp'))