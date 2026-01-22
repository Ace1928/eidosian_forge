from pycadf import cadftaxonomy
from pycadf.helper import api
from pycadf.tests import base
def test_convert_req_action(self):
    self.assertEqual(cadftaxonomy.ACTION_READ, api.convert_req_action('get'))
    self.assertEqual(cadftaxonomy.ACTION_CREATE, api.convert_req_action('POST'))
    self.assertEqual(cadftaxonomy.ACTION_DELETE, api.convert_req_action('deLetE'))