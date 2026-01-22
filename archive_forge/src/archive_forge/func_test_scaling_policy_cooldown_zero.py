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
def test_scaling_policy_cooldown_zero(self):
    now = timeutils.utcnow()
    previous_meta = {'cooldown_end': {now.isoformat(): 'ChangeInCapacity : 1'}, 'scaling_in_progress': False}
    self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
    self.assertIsNone(self.group._check_scaling_allowed(60))