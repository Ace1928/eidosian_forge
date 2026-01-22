from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
Set the ``id`` and ``issued_at`` attributes of a token.

        The process of building a token requires setting attributes about the
        authentication and authorization context, like ``user_id`` and
        ``project_id`` for example. Once a Token object accurately represents
        this information it should be "minted". Tokens are minted when they get
        an ``id`` attribute and their creation time is recorded.

        