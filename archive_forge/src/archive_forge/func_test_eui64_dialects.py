import pickle
import random
from netaddr import (
def test_eui64_dialects():
    mac = EUI('00-1B-77-49-54-FD-12-34')
    assert str(mac) == '00-1B-77-49-54-FD-12-34'
    mac = EUI('00-1B-77-49-54-FD-12-34', dialect=eui64_unix)
    assert str(mac) == '0:1b:77:49:54:fd:12:34'
    mac = EUI('00-1B-77-49-54-FD-12-34', dialect=eui64_unix_expanded)
    assert str(mac) == '00:1b:77:49:54:fd:12:34'
    mac = EUI('00-1B-77-49-54-FD-12-34', dialect=eui64_cisco)
    assert str(mac) == '001b.7749.54fd.1234'
    mac = EUI('00-1B-77-49-54-FD-12-34', dialect=eui64_bare)
    assert str(mac) == '001B774954FD1234'