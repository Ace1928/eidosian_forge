import functools
from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import support
from heat.objects import service as service_objects
from heat.scaling import template as scl_template
Creates a definition object for one of the types in the chain.

        The definition will be built from the given name and type and will
        use the properties specified in the chain's resource_properties
        property. All types in the chain are given the same set of properties.

        :type resource_name: str
        :type resource_type: str
        :param depends_on: if specified, the new resource will depend on the
               resource name specified
        :type depends_on: str
        :return: resource definition suitable for adding to a template
        :rtype: heat.engine.rsrc_defn.ResourceDefinition
        