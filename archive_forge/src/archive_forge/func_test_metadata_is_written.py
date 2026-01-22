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
def test_metadata_is_written(self):
    nowish = timeutils.utcnow()
    reason = 'cool as'
    meta_set = self.patchobject(self.group, 'metadata_set')
    self.patchobject(timeutils, 'utcnow', return_value=nowish)
    self.group._finished_scaling(60, reason)
    cooldown_end = nowish + datetime.timedelta(seconds=60)
    meta_set.assert_called_once_with({'cooldown_end': {cooldown_end.isoformat(): reason}, 'scaling_in_progress': False})