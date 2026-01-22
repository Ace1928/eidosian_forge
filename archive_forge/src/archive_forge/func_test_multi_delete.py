import time
import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_multi_delete(self):
    vol_type1 = uuid.uuid4().hex
    vol_type2 = uuid.uuid4().hex
    self.openstack('volume type create %s' % vol_type1)
    time.sleep(5)
    self.openstack('volume type create %s' % vol_type2)
    time.sleep(5)
    cmd = 'volume type delete %s %s' % (vol_type1, vol_type2)
    raw_output = self.openstack(cmd)
    self.assertOutput('', raw_output)