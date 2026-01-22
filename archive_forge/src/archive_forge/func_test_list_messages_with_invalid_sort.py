from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@ddt.data('fake', 'fake:asc', 'fake:desc')
def test_list_messages_with_invalid_sort(self, sort_string):
    cs = fakes.FakeClient(api_versions.APIVersion('3.5'))
    self.assertRaises(ValueError, cs.messages.list, sort=sort_string)