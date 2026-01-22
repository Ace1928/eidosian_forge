from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@property
def user_domain(self):
    if not self.__user_domain:
        if self.user:
            self.__user_domain = PROVIDERS.resource_api.get_domain(self.user['domain_id'])
    return self.__user_domain