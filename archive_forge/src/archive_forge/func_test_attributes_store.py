import collections
import datetime
import itertools
import json
import os
import sys
from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import attributes
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import clients
from heat.engine import constraints
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import node_data
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources.openstack.heat import none_resource
from heat.engine.resources.openstack.heat import test_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.engine import translation
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_object
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
import neutronclient.common.exceptions as neutron_exp
def test_attributes_store(self):
    res_def = rsrc_defn.ResourceDefinition('test_resource', 'ResWithStringPropAndAttr')
    res = generic_rsrc.ResWithStringPropAndAttr('test_res_attr_store', res_def, self.stack)
    res.action = res.CREATE
    res.status = res.COMPLETE
    res.store()
    res.store_attributes()
    self.assertIsNone(res._attr_data_id)
    with mock.patch.object(res, '_resolve_attribute') as res_attr:
        attr_val = '0123 four'
        res_attr.return_value = attr_val
        res.attributes['string']
        res.store_attributes()
        self.assertIsNotNone(res._attr_data_id)
        self.assertEqual({'string': attr_val}, rpd_object.ResourcePropertiesData.get_by_id(res.context, res._attr_data_id).data)