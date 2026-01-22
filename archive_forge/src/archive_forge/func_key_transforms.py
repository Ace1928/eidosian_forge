from collections import abc
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import opts
from oslo_policy import policy
from glance.common import exception
from glance.domain import proxy
from glance import policies
def key_transforms(self, key):
    transforms = {'id': 'image_id', 'project_id': 'owner', 'member_id': 'member'}
    return transforms.get(key, key)