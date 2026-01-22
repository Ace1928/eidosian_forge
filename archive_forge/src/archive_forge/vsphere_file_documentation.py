from __future__ import absolute_import, division, print_function
import socket
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import PY2
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.urllib.parse import quote, urlencode
from ansible.module_utils.urls import open_url
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
 Constructs a URL path that VSphere accepts reliably 