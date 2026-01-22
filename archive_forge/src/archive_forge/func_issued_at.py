from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@issued_at.setter
def issued_at(self, value):
    if not isinstance(value, str):
        raise ValueError('issued_at must be a string.')
    self.__issued_at = value