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
@db_utils.retry_on_stale_data_error
@context_manager.writer
def user_creds_delete(context, user_creds_id):
    creds = context.session.get(models.UserCreds, user_creds_id)
    if not creds:
        raise exception.NotFound(_('Attempt to delete user creds with id %(id)s that does not exist') % {'id': user_creds_id})
    context.session.delete(creds)