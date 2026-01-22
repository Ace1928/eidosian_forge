import io
import tempfile
import textwrap
import six
from six.moves import configparser
import sys
from pbr.tests import base
from pbr import util
def test_provides_extras(self):
    ini = '\n        [metadata]\n        provides_extras = foo\n                          bar\n        '
    config = config_from_ini(ini)
    kwargs = util.setup_cfg_to_setup_kwargs(config)
    self.assertEqual(['foo', 'bar'], kwargs['provides_extras'])