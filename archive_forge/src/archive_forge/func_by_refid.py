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
def by_refid(name):
    rid = name.replace('id-', '')
    if rid not in self.nested_rsrcs:
        return None
    res = mock.Mock()
    res.name = rid
    return res