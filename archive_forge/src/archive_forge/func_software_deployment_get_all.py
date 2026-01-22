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
def software_deployment_get_all(context, server_id=None):
    sd = models.SoftwareDeployment
    query = context.session.query(sd).order_by(sd.created_at)
    if not context.is_admin:
        query = query.filter(sqlalchemy.or_(sd.tenant == context.tenant_id, sd.stack_user_project_id == context.tenant_id))
    if server_id:
        query = query.filter_by(server_id=server_id)
    query = query.join(models.SoftwareDeployment.config).options(orm.contains_eager(models.SoftwareDeployment.config))
    return query.all()