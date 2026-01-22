import types
import weakref
import six
from apitools.base.protorpclite import util
def validate_default_element(self, value):
    """Validate default element of Enum field.

        Enum fields allow for delayed resolution of default values
        when the type of the field has not been resolved. The default
        value of a field may be a string or an integer. If the Enum
        type of the field has been resolved, the default value is
        validated against that type.

        Args:
          value: Value to validate.

        Raises:
          ValidationError if value is not expected message type.

        """
    if isinstance(value, (six.string_types, six.integer_types)):
        if self.__type:
            self.__type(value)
        return value
    return super(EnumField, self).validate_default_element(value)