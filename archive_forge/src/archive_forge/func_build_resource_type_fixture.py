import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def build_resource_type_fixture(**kwargs):
    resource_type = {'name': 'MyTestResourceType', 'protected': 0}
    resource_type.update(kwargs)
    return resource_type