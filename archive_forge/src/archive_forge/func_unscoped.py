from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@property
def unscoped(self):
    return not any([self.system_scoped, self.domain_scoped, self.project_scoped, self.trust_scoped])