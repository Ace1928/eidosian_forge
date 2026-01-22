from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.capabilities import Capabilities
def test__repr__when_empty(self):
    cap = Capabilities(None, {})
    self.assertEqual('<Capabilities: None>', repr(cap))