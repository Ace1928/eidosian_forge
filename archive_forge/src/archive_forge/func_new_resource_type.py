from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
def new_resource_type(self, namespace, name, **kwargs):
    created_at = timeutils.utcnow()
    updated_at = created_at
    return MetadefResourceType(namespace, name, kwargs.get('prefix'), kwargs.get('properties_target'), created_at, updated_at)