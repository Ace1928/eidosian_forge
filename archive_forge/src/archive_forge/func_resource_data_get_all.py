import datetime
import functools
import itertools
import random
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import orm
from heat.common import crypt
from heat.common import exception
from heat.common.i18n import _
from heat.db import filters as db_filters
from heat.db import models
from heat.db import utils as db_utils
from heat.engine import environment as heat_environment
from heat.rpc import api as rpc_api
@context_manager.reader
def resource_data_get_all(context, resource_id, data=None):
    """Looks up resource_data by resource.id.

    If data is encrypted, this method will decrypt the results.
    """
    if data is None:
        data = context.session.query(models.ResourceData).filter_by(resource_id=resource_id).all()
    if not data:
        raise exception.NotFound(_('no resource data found'))
    ret = {}
    for res in data:
        if res.redact:
            try:
                ret[res.key] = crypt.decrypt(res.decrypt_method, res.value)
                continue
            except exception.InvalidEncryptionKey:
                LOG.exception('Failed to decrypt resource data %(rkey)s for %(rid)s, ignoring.', {'rkey': res.key, 'rid': resource_id})
        ret[res.key] = res.value
    return ret