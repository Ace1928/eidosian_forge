import collections
import contextlib
import datetime as dt
import itertools
import pydoc
import re
import tenacity
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import short_id
from heat.common import timeutils
from heat.engine import attributes
from heat.engine.cfn import template as cfn_tmpl
from heat.engine import clients
from heat.engine.clients import default_client_plugin
from heat.engine import environment
from heat.engine import event
from heat.engine import function
from heat.engine.hot import template as hot_tmpl
from heat.engine import node_data
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import status
from heat.engine import support
from heat.engine import sync_point
from heat.engine import template
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_objects
from heat.objects import resource_properties_data as rpd_objects
from heat.rpc import client as rpc_client
def referenced_attrs(self, stk_defn=None, in_resources=True, in_outputs=True, load_all=False):
    """Return the set of all attributes referenced in the template.

        This enables the resource to calculate which of its attributes will
        be used. By default, attributes referenced in either other resources
        or outputs will be included. Either can be excluded by setting the
        `in_resources` or `in_outputs` parameters to False. To limit to a
        subset of outputs, pass an iterable of the output names to examine
        for the `in_outputs` parameter.

        The set of referenced attributes is calculated from the
        StackDefinition object provided, or from the stack's current
        definition if none is passed.
        """
    if stk_defn is None:
        stk_defn = self.stack.defn

    def get_dep_attrs(source):
        return set(itertools.chain.from_iterable((s.dep_attrs(self.name, load_all) for s in source)))
    refd_attrs = set()
    if in_resources:
        enabled_resources = stk_defn.enabled_rsrc_names()
        refd_attrs |= get_dep_attrs((stk_defn.resource_definition(r_name) for r_name in enabled_resources))
    subset_outputs = isinstance(in_outputs, collections.abc.Iterable)
    if subset_outputs or in_outputs:
        if not subset_outputs:
            in_outputs = stk_defn.enabled_output_names()
        refd_attrs |= get_dep_attrs((stk_defn.output_definition(op_name) for op_name in in_outputs))
    if attributes.ALL_ATTRIBUTES in refd_attrs:
        refd_attrs.remove(attributes.ALL_ATTRIBUTES)
        refd_attrs |= set(self.attributes) - {self.SHOW}
    return refd_attrs