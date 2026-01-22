from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
Display info about playbook statistics