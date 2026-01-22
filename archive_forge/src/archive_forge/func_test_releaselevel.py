import pyomo.common.unittest as unittest
import pyomo.version as pyomo_ver
def test_releaselevel(self):
    _relLevel = pyomo_ver.version_info[3].split('{')[0].strip()
    self.assertIn(_relLevel, ('devel', 'VOTD', 'final'))