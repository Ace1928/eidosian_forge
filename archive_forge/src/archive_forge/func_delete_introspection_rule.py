from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def delete_introspection_rule(self, introspection_rule, ignore_missing=True):
    """Delete an introspection rule.

        :param introspection_rule: The value can be either the ID of an
            introspection rule or a
            :class:`~.introspection_rule.IntrospectionRule` instance.
        :param bool ignore_missing: When set to ``False``, an
            exception:class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the introspection rule could not be found. When set to
            ``True``, no exception will be raised when attempting to delete a
            non-existent introspection rule.

        :returns: ``None``
        """
    self._delete(_introspection_rule.IntrospectionRule, introspection_rule, ignore_missing=ignore_missing)