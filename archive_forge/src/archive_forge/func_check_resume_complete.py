import random
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_utils import timeutils
def check_resume_complete(self, cookie):
    return self._check_status_complete(*cookie)