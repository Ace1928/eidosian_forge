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
def test_password_default_prompts(self):
    self._check_default_password_prompt('FTP %(user)s@%(host)s password: ', 'ftp')
    self._check_default_password_prompt('FTP %(user)s@%(host)s:%(port)d password: ', 'ftp', port=10020)
    self._check_default_password_prompt('SSH %(user)s@%(host)s:%(port)d password: ', 'ssh', port=12345)
    self._check_default_password_prompt('SMTP %(user)s@%(host)s password: ', 'smtp')
    self._check_default_password_prompt('SMTP %(user)s@%(host)s password: ', 'smtp', host='bar.org:10025')
    self._check_default_password_prompt('SMTP %(user)s@%(host)s:%(port)d password: ', 'smtp', port=10025)