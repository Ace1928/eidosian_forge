from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
Return a Template object representing the group's current template.

        Note that this does *not* include any environment data.
        