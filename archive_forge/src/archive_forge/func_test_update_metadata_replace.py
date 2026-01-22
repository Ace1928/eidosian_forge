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
def test_update_metadata_replace(self):
    """Updating the config's metadata causes a config replacement."""
    lc_template = '\n{\n  "AWSTemplateFormatVersion" : "2010-09-09",\n  "Resources": {\n    "JobServerConfig" : {\n      "Type" : "AWS::AutoScaling::LaunchConfiguration",\n      "Metadata": {"foo": "bar"},\n      "Properties": {\n        "ImageId"           : "foo",\n        "InstanceType"      : "m1.large",\n        "KeyName"           : "test",\n      }\n    }\n  }\n}\n'
    self.stub_ImageConstraint_validate()
    self.stub_FlavorConstraint_validate()
    self.stub_KeypairConstraint_validate()
    t = template_format.parse(lc_template)
    stack = utils.parse_stack(t)
    rsrc = self.create_resource(t, stack, 'JobServerConfig')
    props = copy.copy(rsrc.properties.data)
    metadata = copy.copy(rsrc.metadata_get())
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props, metadata)
    scheduler.TaskRunner(rsrc.update, update_snippet)()
    self.assertEqual('bar', metadata['foo'])
    metadata['foo'] = 'wibble'
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props, metadata)
    updater = scheduler.TaskRunner(rsrc.update, update_snippet)
    self.assertRaises(resource.UpdateReplace, updater)