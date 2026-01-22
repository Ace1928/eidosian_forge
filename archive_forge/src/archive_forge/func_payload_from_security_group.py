from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
from ansible.module_utils.basic import AnsibleModule
from uuid import uuid4
def payload_from_security_group(security_group):
    return dict(((k, v) for k, v in security_group.items() if k != 'id' and v is not None))