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
def test_dont_purge_project_shared_raw_template_files(self):
    now = timeutils.utcnow()
    delta = datetime.timedelta(seconds=3600 * 7)
    deleted = [now - delta * i for i in range(1, 6)]
    tmpl_files = [template_files.TemplateFiles({'foo': 'more file contents'}) for i in range(3)]
    [tmpl_file.store(self.ctx) for tmpl_file in tmpl_files]
    templates = [create_raw_template(self.ctx, files_id=tmpl_files[i % 3].files_id) for i in range(5)]
    creds = [create_user_creds(self.ctx) for i in range(5)]
    [create_stack(self.ctx, templates[i], creds[i], deleted_at=deleted[i], tenant=UUID1) for i in range(5)]
    db_api.purge_deleted(age=0, granularity='seconds', project_id=UUID3)
    self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[0].files_id))
    self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[1].files_id))
    self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[2].files_id))
    db_api.purge_deleted(age=15, granularity='hours', project_id=UUID1)
    self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[0].files_id))
    self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[1].files_id))
    self.assertRaises(exception.NotFound, db_api.raw_template_files_get, self.ctx, tmpl_files[2].files_id)