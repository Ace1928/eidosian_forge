import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def test_stat_apache2_dir_depth_1_example(self):
    example = _get_list_dir_apache2_depth_1_allprop()
    self.assertRaises(errors.InvalidHttpResponse, self._extract_stat_from_str, example)