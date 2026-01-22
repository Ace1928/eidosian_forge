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
def resource_purge_deleted(context, stack_id):
    filters = {'stack_id': stack_id, 'action': 'DELETE', 'status': 'COMPLETE'}
    query = context.session.query(models.Resource)
    result = query.filter_by(**filters)
    attr_ids = [r.attr_data_id for r in result if r.attr_data_id is not None]
    result.delete()
    if attr_ids:
        context.session.query(models.ResourcePropertiesData).filter(models.ResourcePropertiesData.id.in_(attr_ids)).delete(synchronize_session=False)