from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def introspection_rules(self, **query):
    """Retrieve a generator of introspection rules.

        :param dict query: Optional query parameters to be sent to restrict
            the records to be returned. Available parameters include:

            * ``uuid``: The UUID of the Ironic Inspector rule.
            * ``limit``: List of a logic statementd or operations in rules,
                         that can be evaluated as True or False.
            * ``actions``: List of operations that will be performed
                           if conditions of this rule are fulfilled.
            * ``description``: Rule human-readable description.
            * ``scope``: Scope of an introspection rule. If set, the rule
                         is only applied to nodes that have
                         matching inspection_scope property.

        :returns: A generator of
            :class:`~.introspection_rule.IntrospectionRule`
            objects
        """
    return self._list(_introspection_rule.IntrospectionRule, **query)