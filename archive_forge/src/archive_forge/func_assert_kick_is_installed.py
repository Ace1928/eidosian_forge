from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
def assert_kick_is_installed(module):
    if not HAS_KICK:
        module.fail_json(msg='Firepower-kickstart library is required to run this module. Please, install the library with `pip install firepower-kickstart` command and run the playbook again.')