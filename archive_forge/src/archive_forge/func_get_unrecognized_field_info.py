import types
import weakref
import six
from apitools.base.protorpclite import util
def get_unrecognized_field_info(self, key, value_default=None, variant_default=None):
    """Get the value and variant of an unknown field in this message.

        Args:
          key: The name or number of the field to retrieve.
          value_default: Value to be returned if the key isn't found.
          variant_default: Value to be returned as variant if the key isn't
            found.

        Returns:
          (value, variant), where value and variant are whatever was passed
          to set_unrecognized_field.
        """
    value, variant = self.__unrecognized_fields.get(key, (value_default, variant_default))
    return (value, variant)