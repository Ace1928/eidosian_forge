import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def build_task_fixtures(self):
    self.context.project_id = str(uuid.uuid4())
    fixtures = [{'owner': self.context.owner, 'type': 'import', 'input': {'import_from': 'file:///a.img', 'import_from_format': 'qcow2', 'image_properties': {'name': 'GreatStack 1.22', 'tags': ['lamp', 'custom']}}}, {'owner': self.context.owner, 'type': 'import', 'input': {'import_from': 'file:///b.img', 'import_from_format': 'qcow2', 'image_properties': {'name': 'GreatStack 1.23', 'tags': ['lamp', 'good']}}}, {'owner': self.context.owner, 'type': 'export', 'input': {'export_uuid': 'deadbeef-dead-dead-dead-beefbeefbeef', 'export_to': 'swift://cloud.foo/myaccount/mycontainer/path', 'export_format': 'qcow2'}}]
    return [build_task_fixture(**fixture) for fixture in fixtures]