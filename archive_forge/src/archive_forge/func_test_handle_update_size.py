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
def test_handle_update_size(self):
    self.instance_group._try_rolling_update = mock.Mock(return_value=None)
    self.instance_group.resize = mock.Mock(return_value=None)
    props = {'Size': 5}
    defn = rsrc_defn.ResourceDefinition('nopayload', 'AWS::AutoScaling::AutoScalingGroup', props)
    self.instance_group.handle_update(defn, None, props)
    self.instance_group.resize.assert_called_once_with(5)