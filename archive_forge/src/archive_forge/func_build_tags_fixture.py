import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def build_tags_fixture(tag_name_list):
    tag_list = []
    for tag_name in tag_name_list:
        tag_list.append({'name': tag_name})
    return tag_list