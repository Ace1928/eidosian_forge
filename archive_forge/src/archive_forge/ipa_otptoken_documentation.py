from __future__ import absolute_import, division, print_function
import base64
import traceback
from ansible.module_utils.basic import AnsibleModule, sanitize_keys
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
Checks to see if the requested modifications are valid.  Some elements
       cannot be modified after initial creation.  However, we still want to
       validate arguments that are specified, but are not different than what
       is currently set on the server.
    