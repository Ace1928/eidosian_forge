from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import group_types
def test_list_group_types_pre_version(self):
    self.assertRaises(exc.VersionNotFoundForAPIMethod, pre_cs.group_types.list)