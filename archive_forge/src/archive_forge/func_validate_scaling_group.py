import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def validate_scaling_group(self, t, stack, resource_name):
    conf = stack['LaunchConfig']
    self.assertIsNone(conf.validate())
    scheduler.TaskRunner(conf.create)()
    self.assertEqual((conf.CREATE, conf.COMPLETE), conf.state)
    rsrc = stack[resource_name]
    self.assertIsNone(rsrc.validate())
    return rsrc