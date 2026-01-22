import random
import string
import uuid
import warnings
import fixtures
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
import sqlalchemy
from sqlalchemy import exc as sqla_exc
from heat.common import context
from heat.db import api as db_api
from heat.db import models
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine import stack
from heat.engine import template
def dummy_system_admin_context():
    """Return a heat.common.context.RequestContext for system-admin.

    :returns: an instance of heat.common.context.RequestContext

    """
    ctx = dummy_context(roles=['admin', 'member', 'reader'])
    ctx.system_scope = 'all'
    ctx.project_id = None
    ctx.tenant_id = None
    return ctx