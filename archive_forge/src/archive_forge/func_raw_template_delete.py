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
@context_manager.writer
def raw_template_delete(context, template_id):
    try:
        raw_template = _raw_template_get(context, template_id)
    except exception.NotFound:
        return
    raw_tmpl_files_id = raw_template.files_id
    context.session.delete(raw_template)
    if raw_tmpl_files_id is None:
        return
    if context.session.query(models.RawTemplate).filter_by(files_id=raw_tmpl_files_id).first() is None:
        try:
            raw_tmpl_files = _raw_template_files_get(context, raw_tmpl_files_id)
        except exception.NotFound:
            return
        context.session.delete(raw_tmpl_files)