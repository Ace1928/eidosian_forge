import io
import tempfile
import textwrap
import six
from six.moves import configparser
import sys
from pbr.tests import base
from pbr import util
def test_invalid_marker_raises_error(self):
    config = {'extras': {'test': "foo :bad_marker>'1.0'"}}
    self.assertRaises(SyntaxError, util.setup_cfg_to_setup_kwargs, config)