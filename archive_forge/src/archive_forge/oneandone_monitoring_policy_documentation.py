from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (

    Removes a monitoring policy.

    module : AnsibleModule object
    oneandone_conn: authenticated oneandone object
    