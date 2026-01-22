from __future__ import absolute_import, division, print_function
import os
import re
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def install_command(self):
    cmd = 'tmsh load sys ucs /var/local/ucs/{0}'.format(self.basename)
    options = OrderedDict(sorted(self.options.items(), key=lambda t: t[0]))
    for k, v in iteritems(options):
        if v is False or v is None:
            continue
        elif k == 'passphrase':
            cmd += ' %s %s' % (k, v)
        else:
            cmd += ' %s' % k
    return cmd