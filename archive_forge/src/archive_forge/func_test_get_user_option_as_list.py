import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def test_get_user_option_as_list(self):
    conf, parser = self.make_config_parser('\na_list = a,b,c\nlength_1 = 1,\none_item = x\n')
    get_list = conf.get_user_option_as_list
    self.assertEqual(['a', 'b', 'c'], get_list('a_list'))
    self.assertEqual(['1'], get_list('length_1'))
    self.assertEqual('x', conf.get_user_option('one_item'))
    self.assertEqual(['x'], get_list('one_item'))