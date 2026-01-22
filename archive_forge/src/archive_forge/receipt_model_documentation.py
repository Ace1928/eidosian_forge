from oslo_log import log
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.auth import core
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.identity.backends import resource_options as ro
Set the ``id`` and ``issued_at`` attributes of a receipt.

        The process of building a Receipt requires setting attributes about the
        partial authentication context, like ``user_id`` and ``methods`` for
        example. Once a Receipt object accurately represents this information
        it should be "minted". Receipt are minted when they get an ``id``
        attribute and their creation time is recorded.
        