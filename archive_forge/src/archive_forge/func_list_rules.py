from oslo_log import versionutils
from oslo_policy import policy
from keystone.common.policies import base
def list_rules():
    return region_policies