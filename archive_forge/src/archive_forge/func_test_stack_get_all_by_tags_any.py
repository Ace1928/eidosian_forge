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
def test_stack_get_all_by_tags_any(self):
    stacks = [self._setup_test_stack('stacks_tags_any_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
    stacks[0].tags = ['tag2']
    stacks[0].store()
    stacks[1].tags = ['tag1', 'tag2']
    stacks[1].store()
    stacks[2].tags = ['tag1', 'tag3']
    stacks[2].store()
    st_db = db_api.stack_get_all(self.ctx, tags_any=['tag1'])
    self.assertEqual(2, len(st_db))
    st_db = db_api.stack_get_all(self.ctx, tags_any=['tag1', 'tag2', 'tag3'])
    self.assertEqual(3, len(st_db))