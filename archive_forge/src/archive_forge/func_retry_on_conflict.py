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
def retry_on_conflict(func):
    wrapper = tenacity.retry(stop=tenacity.stop_after_attempt(11), wait=tenacity.wait_random_exponential(multiplier=0.5, max=60), retry=tenacity.retry_if_exception_type(exception.ConcurrentTransaction), reraise=True)
    return wrapper(func)