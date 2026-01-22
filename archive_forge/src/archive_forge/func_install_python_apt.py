from __future__ import absolute_import, division, print_function
import copy
import glob
import json
import os
import re
import sys
import tempfile
import random
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.locale import get_best_parsable_locale
def install_python_apt(module, apt_pkg_name):
    if not module.check_mode:
        apt_get_path = module.get_bin_path('apt-get')
        if apt_get_path:
            rc, so, se = module.run_command([apt_get_path, 'update'])
            if rc != 0:
                module.fail_json(msg="Failed to auto-install %s. Error was: '%s'" % (apt_pkg_name, se.strip()))
            rc, so, se = module.run_command([apt_get_path, 'install', apt_pkg_name, '-y', '-q'])
            if rc != 0:
                module.fail_json(msg="Failed to auto-install %s. Error was: '%s'" % (apt_pkg_name, se.strip()))
    else:
        module.fail_json(msg='%s must be installed to use check mode' % apt_pkg_name)