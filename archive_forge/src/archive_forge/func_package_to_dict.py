from __future__ import absolute_import, division, print_function
import os
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
def package_to_dict(package):
    return {'nevra': package.get_nevra(), 'envra': package.get_nevra(), 'name': package.get_name(), 'arch': package.get_arch(), 'epoch': str(package.get_epoch()), 'release': package.get_release(), 'version': package.get_version(), 'repo': package.get_repo_id(), 'yumstate': 'installed' if package.is_installed() else 'available'}