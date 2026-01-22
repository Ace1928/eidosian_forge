import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@is_admin_project.deleter
def is_admin_project(self):
    self.root.pop('is_admin_project', None)