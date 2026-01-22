import os
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import nested_structure_coder
def set_save_dataset_attributes(dataset, shard_func, path):
    """Sets parameters for SaveDatasetOp and SaveDatasetV2Op."""
    if shard_func is None:
        use_shard_func = False
        shard_func = lambda *x: None
    else:
        use_shard_func = True
    wrapped_func = structured_function.StructuredFunctionWrapper(shard_func, 'save()', input_structure=dataset.element_spec, add_to_graph=False)
    encoded = nested_structure_coder.encode_structure(dataset.element_spec)
    gfile.MakeDirs(path)
    with gfile.GFile(os.path.join(path, dataset_ops.DATASET_SPEC_FILENAME), 'wb') as f:
        f.write(encoded.SerializeToString())
    path = ops.convert_to_tensor(path, dtype=dtypes.string, name='path')
    shard_func = wrapped_func.function
    shard_func.add_to_graph(ops.get_default_graph())
    dataset._apply_debug_options()
    return (dataset, shard_func, use_shard_func, path)