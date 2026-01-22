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
@retry_on_db_error
@context_manager.writer
def sync_point_update_input_data(context, entity_id, traversal_id, is_update, atomic_key, input_data):
    entity_id = str(entity_id)
    rows_updated = context.session.query(models.SyncPoint).filter_by(entity_id=entity_id, traversal_id=traversal_id, is_update=is_update, atomic_key=atomic_key).update({'input_data': input_data, 'atomic_key': atomic_key + 1})
    return rows_updated