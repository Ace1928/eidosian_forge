import collections
from oslo_config import cfg
from oslo_log import log as logging
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
import tenacity
from heat.common import crypt
from heat.common import exception
from heat.common.i18n import _
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
from heat.objects import resource_data
from heat.objects import resource_properties_data as rpd
@property
def properties_data(self):
    if not self._properties_data and self.rsrc_prop_data_id is not None:
        LOG.info('rsrc_prop_data lazy load')
        rpd_obj = rpd.ResourcePropertiesData.get_by_id(self._context, self.rsrc_prop_data_id)
        self._properties_data = rpd_obj.data or {}
    return self._properties_data