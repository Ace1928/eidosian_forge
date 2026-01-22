import types
import weakref
import six
from apitools.base.protorpclite import util
def outer_definition_name(cls):
    """Helper method for creating outer definition name.

        Returns:
          If definition is nested, will return the outer definitions
          name, else the package name.

        """
    outer_definition = cls.message_definition()
    if not outer_definition:
        return util.get_package_for_module(cls.__module__)
    return outer_definition.definition_name()