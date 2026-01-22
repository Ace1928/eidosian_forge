import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.nest.yield_flat_paths', v1=[])
def yield_flat_paths(nest, expand_composites=False):
    """Yields paths for some nested structure.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure.

  Paths are lists of objects which can be str-converted, which may include
  integers or other types which are used as indices in a dict.

  The flat list will be in the corresponding order as if you called
  `nest.flatten` on the structure. This is handy for naming Tensors such
  the TF scope structure matches the tuple structure.

  E.g. if we have a tuple `value = Foo(a=3, b=Bar(c=23, d=42))`

  ```shell
  nest.flatten(value)
  [3, 23, 42]
  list(nest.yield_flat_paths(value))
  [('a',), ('b', 'c'), ('b', 'd')]
  ```

  ```shell
  list(nest.yield_flat_paths({'a': [3]}))
  [('a', 0)]
  list(nest.yield_flat_paths({'a': 3}))
  [('a',)]
  ```

  Args:
    nest: the value to produce a flattened paths list for.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Yields:
    Tuples containing index or key values which form the path to a specific
    leaf value in the nested structure.
  """
    is_nested_fn = _is_nested_or_composite if expand_composites else is_nested
    for k, _ in nest_util.yield_flat_up_to(nest_util.Modality.CORE, nest, nest, is_nested_fn):
        yield k