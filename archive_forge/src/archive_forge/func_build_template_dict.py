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
@staticmethod
def build_template_dict(res_name, res_type, tmpl_type, params, props, outputs, description):
    if tmpl_type == 'hot':
        tmpl_dict = {hot_tmpl.HOTemplate20161014.VERSION: '2016-10-14', hot_tmpl.HOTemplate20161014.DESCRIPTION: description, hot_tmpl.HOTemplate20161014.PARAMETERS: params, hot_tmpl.HOTemplate20161014.OUTPUTS: outputs, hot_tmpl.HOTemplate20161014.RESOURCES: {res_name: {hot_tmpl.HOTemplate20161014.RES_TYPE: res_type, hot_tmpl.HOTemplate20161014.RES_PROPERTIES: props}}}
    else:
        tmpl_dict = {cfn_tmpl.CfnTemplate.ALTERNATE_VERSION: '2012-12-12', cfn_tmpl.CfnTemplate.DESCRIPTION: description, cfn_tmpl.CfnTemplate.PARAMETERS: params, cfn_tmpl.CfnTemplate.RESOURCES: {res_name: {cfn_tmpl.CfnTemplate.RES_TYPE: res_type, cfn_tmpl.CfnTemplate.RES_PROPERTIES: props}}, cfn_tmpl.CfnTemplate.OUTPUTS: outputs}
    return tmpl_dict