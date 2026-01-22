import pickle
import random
from netaddr import (
def test_eui_custom_dialect():

    class mac_custom(mac_unix):
        word_fmt = '%.2X'
    mac = EUI('00-1B-77-49-54-FD', dialect=mac_custom)
    assert str(mac) == '00:1B:77:49:54:FD'