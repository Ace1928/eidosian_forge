from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_REGIONS, SCALEWAY_ENDPOINT, scaleway_argument_spec, Scaleway
def lb_attributes_should_be_changed(target_lb, wished_lb):
    diff = dict(((attr, wished_lb[attr]) for attr in MUTABLE_ATTRIBUTES if target_lb[attr] != wished_lb[attr]))
    if diff:
        return dict(((attr, wished_lb[attr]) for attr in MUTABLE_ATTRIBUTES))
    else:
        return diff