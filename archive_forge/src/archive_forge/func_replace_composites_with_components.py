import abc
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def replace_composites_with_components(structure):
    """Recursively replaces CompositeTensors with their components.

  Args:
    structure: A `nest`-compatible structure, possibly containing composite
      tensors.

  Returns:
    A copy of `structure`, where each composite tensor has been replaced by
    its components.  The result will contain no composite tensors.
    Note that `nest.flatten(replace_composites_with_components(structure))`
    returns the same value as `nest.flatten(structure)`.
  """
    if isinstance(structure, CompositeTensor):
        return replace_composites_with_components(structure._type_spec._to_components(structure))
    elif not nest.is_nested(structure):
        return structure
    else:
        return nest.map_structure(replace_composites_with_components, structure, expand_composites=False)