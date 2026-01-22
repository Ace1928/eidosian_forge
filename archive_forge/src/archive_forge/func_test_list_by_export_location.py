from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_instances
@ddt.data(('id', 'b4991315-eb7d-43ec-979e-5715d4399827'), ('path', '//0.0.0.0/fake_path'))
@ddt.unpack
def test_list_by_export_location(self, filter_type, value):
    cs.share_instances.list(export_location=value)
    cs.assert_called('GET', '/share_instances?export_location_' + filter_type + '=' + value)