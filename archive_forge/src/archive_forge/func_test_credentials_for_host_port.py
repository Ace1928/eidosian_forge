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
def test_credentials_for_host_port(self):
    conf = config.AuthenticationConfig(_file=BytesIO(b'# Identity on foo.net\n[ftp definition]\nscheme=ftp\nport=10021\nhost=foo.net\nuser=joe\npassword=secret-pass\n'))
    self._got_user_passwd('joe', 'secret-pass', conf, 'ftp', 'foo.net', port=10021)
    self._got_user_passwd(None, None, conf, 'ftp', 'foo.net')