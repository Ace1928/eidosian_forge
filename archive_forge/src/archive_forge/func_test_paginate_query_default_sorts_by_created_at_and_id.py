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
@mock.patch.object(db_api.utils, 'paginate_query')
def test_paginate_query_default_sorts_by_created_at_and_id(self, mock_paginate_query):
    query = mock.Mock()
    model = mock.Mock()
    db_api._paginate_query(self.ctx, query, model, sort_keys=None)
    args, _ = mock_paginate_query.call_args
    self.assertIn(['created_at', 'id'], args)