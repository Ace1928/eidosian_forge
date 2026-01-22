from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
A resource that stores network information for share servers.

    Stores network information that will be used by share servers,
    where shares are hosted.
    