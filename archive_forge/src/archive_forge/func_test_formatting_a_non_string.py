import uuid
from keystone.common import utils
from keystone import exception
from keystone.tests import unit
def test_formatting_a_non_string(self):

    def _test(url_template):
        self.assertRaises(exception.MalformedEndpoint, utils.format_url, url_template, {})
    _test(None)
    _test(object())