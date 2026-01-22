import collections
import json
import os
from unittest import mock
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources import template_resource
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_get_os_no_match(self):
    env_str = {'resource_registry': {'resources': {'jerry': {'OS::ResourceType': 'myCloud::ResourceType'}}}}
    env = environment.Environment(env_str)
    cls = env.get_class('GenericResourceType', 'fred')
    self.assertEqual(generic_rsrc.GenericResource, cls)