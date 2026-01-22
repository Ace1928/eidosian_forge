from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import images as api_images
from saharaclient.osc.v1 import images as osc_images
from saharaclient.tests.unit.osc.v1 import test_images as images_v1
def test_image_unregister_no_options(self):
    arglist = []
    verifylist = []
    self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)