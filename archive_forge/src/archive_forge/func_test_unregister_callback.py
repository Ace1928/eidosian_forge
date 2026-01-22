from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
def test_unregister_callback():
    central.manage.unregister_callback()
    assert central._callback is None