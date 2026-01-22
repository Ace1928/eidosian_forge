from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.util import compat
def parse_func_attrs(attributes, allowlist=None):
    """Convert the keyword arguments into function_def attributes.

  Currently only support primitive types: bool, int, float and string.

  Args:
    attributes: the dictionary of attributes.
    allowlist: set of attribute names allowed.
  Returns:
    A dict of attributes where the key is the name of attribute and the value
      is the AttrValue proto.
  Raises:
    ValueError: If the kwargs contains unallowlisted name or unsupported value
      types.
  """
    if not allowlist:
        allowlist = MONOMORPHIC_FUNCTION_ALLOWLIST
    attrs = {}
    for key, value in attributes.items():
        if key not in allowlist:
            raise ValueError(f'Allowlist does not support `{key}` as an attribute.')
        attrs[key] = _parse_func_attr_value(key, value)
    return attrs