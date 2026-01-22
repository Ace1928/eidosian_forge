import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
def map_structure_with_tuple_paths_up_to(shallow_tree, func, *inputs, **kwargs):
    """Applies a function or op to a number of partially flattened inputs.

  Like map_structure_up_to(), except that the 'func' argument takes a path
  tuple as its first argument, followed by the corresponding values from
  *inputs.

  Example:

  ```python
  lowercase = {'a': 'a', 'b': ('b0', 'b1')}
  uppercase = {'a': 'A', 'b': ('B0', 'B1')}

  def print_path_and_values(path, *values):
    print("path: {}, values: {}".format(path, values))

  shallow_tree = {'a': None}
  map_structure_with_tuple_paths_up_to(shallow_tree,
                                       print_path_and_values,
                                       lowercase,
                                       uppercase)
  path: ('a',), values: ('a', 'A')
  path: ('b', 0), values: ('b0', 'B0')
  path: ('b', 1), values: ('b1', 'B1')

  shallow_tree = {'b': None}
  map_structure_with_tuple_paths_up_to(shallow_tree,
                                       print_path_and_values,
                                       lowercase,
                                       uppercase,
                                       check_types=False)
  path: ('b', 1), values: (('bo', 'b1'), ('B0', 'B1'))

  shallow_tree = {'a': None, 'b': {1: None}}
  map_structure_with_tuple_paths_up_to(shallow_tree,
                                       print_path_and_values,
                                       lowercase,
                                       uppercase,
                                       check_types=False)
  path: ('a',), values: ('a', 'A')
  path: ('b', 1), values: ('b1', B1')
  ```

  Args:
    shallow_tree: a shallow structure, common to all the inputs.
    func: callable that takes args (path, inputs_0_value, ... , inputs_N_value),
      where path is a tuple path to an atom in shallow_tree, and inputs_i_value
      is the corresponding value from inputs[i].
    *inputs: structures that are all structurally compatible with shallow_tree.
    **kwargs: kwargs to feed to func(). Special kwarg `check_types` is not
      passed to func, but instead determines whether the types of iterables
      within the structures have to be same (e.g. `map_structure(func, [1],
      (1,))` raises a `TypeError` exception). To allow this set this argument to
      `False`.

  Raises:
    TypeError: If `shallow_tree` is a nested structure but one of `*inputs` is
      not.
    TypeError: If the structure types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the structure lengths of `shallow_tree` are different from
      `input_tree`.

  Returns:
    Result of repeatedly applying `func`. Has the same structure layout as
    `shallow_tree`.
  """
    return nest_util.map_structure_up_to(nest_util.Modality.CORE, shallow_tree, func, *inputs, **kwargs)