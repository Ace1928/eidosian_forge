import datetime
import json
import time
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_utils import timeutils
from sqlalchemy import orm
from sqlalchemy.orm import exc
from sqlalchemy.orm import session
from heat.common import context
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import resource as rsrc
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.engine import template_files
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def create_sync_point(ctx, **kwargs):
    values = {'entity_id': '0782c463-064a-468d-98fd-442efb638e3a', 'is_update': True, 'traversal_id': '899ff81e-fc1f-41f9-f41d-ad1ea7f31d19', 'atomic_key': 0, 'stack_id': 'f6359498-764b-49e7-a515-ad31cbef885b', 'input_data': {}}
    values.update(kwargs)
    return db_api.sync_point_create(ctx, values)