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
def user_creds_create(context):
    values = context.to_dict()
    user_creds_ref = models.UserCreds()
    if values.get('trust_id'):
        method, trust_id = crypt.encrypt(values.get('trust_id'))
        user_creds_ref.trust_id = trust_id
        user_creds_ref.decrypt_method = method
        user_creds_ref.trustor_user_id = values.get('trustor_user_id')
        user_creds_ref.username = None
        user_creds_ref.password = None
        user_creds_ref.tenant = values.get('tenant')
        user_creds_ref.tenant_id = values.get('tenant_id')
        user_creds_ref.auth_url = values.get('auth_url')
        user_creds_ref.region_name = values.get('region_name')
    else:
        user_creds_ref.update(values)
        method, password = crypt.encrypt(values['password'])
        if len(str(password)) > 255:
            raise exception.Error(_('Length of OS_PASSWORD after encryption exceeds Heat limit (255 chars)'))
        user_creds_ref.password = password
        user_creds_ref.decrypt_method = method
    user_creds_ref.save(context.session)
    result = dict(user_creds_ref)
    if values.get('trust_id'):
        result['trust_id'] = values.get('trust_id')
    else:
        result['password'] = values.get('password')
    return result