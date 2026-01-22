import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def tmpl_with_bad_updt_policy():
    t = copy.deepcopy(template)
    rg = t['resources']['group1']
    rg['update_policy'] = {'foo': {}}
    return t