from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def create_introspection_rule(self, **attrs):
    """Create a new introspection rules from attributes.

        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~.introspection_rule.IntrospectionRule`,
            comprised of the properties on the IntrospectionRule class.

        :returns: :class:`~.introspection_rule.IntrospectionRule` instance.
        """
    return self._create(_introspection_rule.IntrospectionRule, **attrs)