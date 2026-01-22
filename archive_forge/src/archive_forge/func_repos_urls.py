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
@property
def repos_urls(self):
    _repositories = []
    for parsed_repos in self.files.values():
        for parsed_repo in parsed_repos:
            valid = parsed_repo[1]
            enabled = parsed_repo[2]
            source_line = parsed_repo[3]
            if not valid or not enabled:
                continue
            if source_line.startswith('ppa:'):
                source, ppa_owner, ppa_name = self._expand_ppa(source_line)
                _repositories.append(source)
            else:
                _repositories.append(source_line)
    return _repositories