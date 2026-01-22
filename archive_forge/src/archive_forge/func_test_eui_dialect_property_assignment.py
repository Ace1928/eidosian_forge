import pickle
import random
from netaddr import (
def test_eui_dialect_property_assignment():
    mac = EUI('00-1B-77-49-54-FD')
    assert str(mac) == '00-1B-77-49-54-FD'
    mac.dialect = mac_pgsql
    assert str(mac) == '001b77:4954fd'