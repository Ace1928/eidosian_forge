import pickle
import random
from netaddr import (
def test_oui_hash():
    oui0 = OUI(0)
    oui1 = OUI(1)
    oui_dict = {oui0: None, oui1: None}
    assert list(oui_dict.keys()) == [OUI(0), OUI(1)]