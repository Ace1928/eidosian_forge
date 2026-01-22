from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def test_keystore(module, keystore_path):
    """ Check if we can access keystore as file or not """
    if keystore_path is None:
        keystore_path = ''
    if not os.path.exists(keystore_path) and (not os.path.isfile(keystore_path)):
        module.fail_json(changed=False, msg="Module require existing keystore at keystore_path '%s'" % keystore_path)