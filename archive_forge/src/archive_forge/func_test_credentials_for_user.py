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
def test_credentials_for_user(self):
    conf = config.AuthenticationConfig(_file=BytesIO(b'\n[with user]\nscheme=http\nhost=bar.org\nuser=jim\npassword=jimpass\n'))
    self._got_user_passwd('jim', 'jimpass', conf, 'http', 'bar.org')
    self._got_user_passwd('jim', 'jimpass', conf, 'http', 'bar.org', user='jim')
    self._got_user_passwd(None, None, conf, 'http', 'bar.org', user='georges')