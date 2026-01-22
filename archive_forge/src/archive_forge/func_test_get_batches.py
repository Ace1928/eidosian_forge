import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_get_batches(self):
    batches = list(instgrp.InstanceGroup._get_batches(self.tgt_cap, self.curr_cap, self.bat_size, self.min_serv))
    self.assertEqual(self.batches, batches)