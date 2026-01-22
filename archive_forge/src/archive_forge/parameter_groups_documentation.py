from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
Validate the parameter group.

        Validate that each parameter belongs to only one Parameter Group and
        that each parameter name in the group references a valid parameter.
        