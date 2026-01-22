import types
import weakref
import six
from apitools.base.protorpclite import util
def get_assigned_value(self, name):
    """Get the assigned value of an attribute.

        Get the underlying value of an attribute. If value has not
        been set, will not return the default for the field.

        Args:
          name: Name of attribute to get.

        Returns:
          Value of attribute, None if it has not been set.

        """
    message_type = type(self)
    try:
        field = message_type.field_by_name(name)
    except KeyError:
        raise AttributeError('Message %s has no field %s' % (message_type.__name__, name))
    return self.__tags.get(field.number)