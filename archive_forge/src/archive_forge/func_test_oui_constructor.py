import pickle
import random
from netaddr import (
def test_oui_constructor():
    oui = OUI(524336)
    assert str(oui) == '08-00-30'
    assert oui == OUI('08-00-30')
    assert oui.registration(0).address == ['2380 N. ROSE AVENUE', 'OXNARD  CA  93010', 'US']
    assert oui.registration(0).org == 'NETWORK RESEARCH CORPORATION'
    assert oui.registration(0).oui == '08-00-30'
    assert oui.registration(1).address == ['GPO BOX 2476V', 'MELBOURNE  VIC  3001', 'AU']
    assert oui.registration(1).org == 'ROYAL MELBOURNE INST OF TECH'
    assert oui.registration(1).oui == '08-00-30'
    assert oui.registration(2).address == ['CH-1211', 'GENEVE  SUISSE/SWITZ  023', 'CH']
    assert oui.registration(2).org == 'CERN'
    assert oui.registration(2).oui == '08-00-30'
    assert oui.reg_count == 3