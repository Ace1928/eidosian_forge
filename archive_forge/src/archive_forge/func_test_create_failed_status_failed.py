from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_create_failed_status_failed(self):
    self._create_failed_bad_status('FAILED', 'Went to status FAILED due to "The database instance was created, but heat failed to set up the datastore. If a database instance is in the FAILED state, it should be deleted and a new one should be created."')