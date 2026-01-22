import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_no_instance_list(self):
    """Tests inheritance of InstanceList attribute.

        The InstanceList attribute is not inherited from
        AutoScalingResourceGroup's superclasses.
        """
    self.assertRaises(exception.InvalidTemplateAttribute, self.group.FnGetAtt, 'InstanceList')