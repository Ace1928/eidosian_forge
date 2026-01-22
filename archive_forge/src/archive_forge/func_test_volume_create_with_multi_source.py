from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume
def test_volume_create_with_multi_source(self):
    arglist = ['--image', 'source_image', '--source', 'source_volume', '--snapshot', 'source_snapshot', '--size', str(self.new_volume.size), self.new_volume.name]
    verifylist = [('image', 'source_image'), ('source', 'source_volume'), ('snapshot', 'source_snapshot'), ('size', self.new_volume.size), ('name', self.new_volume.name)]
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)