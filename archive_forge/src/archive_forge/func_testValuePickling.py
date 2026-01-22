import datetime
import pickle
import sys
from copy import deepcopy
from tests.base import BaseTestCase
from pyasn1.type import useful
def testValuePickling(self):
    old_asn1 = useful.UTCTime('170711000102')
    serialised = pickle.dumps(old_asn1)
    assert serialised
    new_asn1 = pickle.loads(serialised)
    assert new_asn1 == old_asn1