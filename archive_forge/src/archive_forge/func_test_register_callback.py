from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
def test_register_callback():
    local_notified = []
    central.manage.register_callback(notify)
    for element in central.select('/projects'):
        local_notified.append(element)
    assert notified == local_notified