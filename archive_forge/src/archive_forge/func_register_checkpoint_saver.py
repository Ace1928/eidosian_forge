import collections
import re
from tensorflow.python.util import tf_inspect
def register_checkpoint_saver(package='Custom', name=None, predicate=None, save_fn=None, restore_fn=None, strict_predicate_restore=True):
    """Registers functions which checkpoints & restores objects with custom steps.

  If you have a class that requires complicated coordination between multiple
  objects when checkpointing, then you will need to register a custom saver
  and restore function. An example of this is a custom Variable class that
  splits the variable across different objects and devices, and needs to write
  checkpoints that are compatible with different configurations of devices.

  The registered save and restore functions are used in checkpoints and
  SavedModel.

  Please make sure you are familiar with the concepts in the [Checkpointing
  guide](https://www.tensorflow.org/guide/checkpoint), and ops used to save the
  V2 checkpoint format:

  * io_ops.SaveV2
  * io_ops.MergeV2Checkpoints
  * io_ops.RestoreV2

  **Predicate**

  The predicate is a filter that will run on every `Trackable` object connected
  to the root object. This function determines whether a `Trackable` should use
  the registered functions.

  Example: `lambda x: isinstance(x, CustomClass)`

  **Custom save function**

  This is how checkpoint saving works normally:
  1. Gather all of the Trackables with saveable values.
  2. For each Trackable, gather all of the saveable tensors.
  3. Save checkpoint shards (grouping tensors by device) with SaveV2
  4. Merge the shards with MergeCheckpointV2. This combines all of the shard's
     metadata, and renames them to follow the standard shard pattern.

  When a saver is registered, Trackables that pass the registered `predicate`
  are automatically marked as having saveable values. Next, the custom save
  function replaces steps 2 and 3 of the saving process. Finally, the shards
  returned by the custom save function are merged with the other shards.

  The save function takes in a dictionary of `Trackables` and a `file_prefix`
  string. The function should save checkpoint shards using the SaveV2 op, and
  list of the shard prefixes. SaveV2 is currently required to work a correctly,
  because the code merges all of the returned shards, and the `restore_fn` will
  only be given the prefix of the merged checkpoint. If you need to be able to
  save and restore from unmerged shards, please file a feature request.

  Specification and example of the save function:

  ```
  def save_fn(trackables, file_prefix):
    # trackables: A dictionary mapping unique string identifiers to trackables
    # file_prefix: A unique file prefix generated using the registered name.
    ...
    # Gather the tensors to save.
    ...
    io_ops.SaveV2(file_prefix, tensor_names, shapes_and_slices, tensors)
    return file_prefix  # Returns a tensor or a list of string tensors
  ```

  The save function is executed before the unregistered save ops.

  **Custom restore function**

  Normal checkpoint restore behavior:
  1. Gather all of the Trackables that have saveable values.
  2. For each Trackable, get the names of the desired tensors to extract from
     the checkpoint.
  3. Use RestoreV2 to read the saved values, and pass the restored tensors to
     the corresponding Trackables.

  The custom restore function replaces steps 2 and 3.

  The restore function also takes a dictionary of `Trackables` and a
  `merged_prefix` string. The `merged_prefix` is different from the
  `file_prefix`, since it contains the renamed shard paths. To read from the
  merged checkpoint, you must use `RestoreV2(merged_prefix, ...)`.

  Specification:

  ```
  def restore_fn(trackables, merged_prefix):
    # trackables: A dictionary mapping unique string identifiers to Trackables
    # merged_prefix: File prefix of the merged shard names.

    restored_tensors = io_ops.restore_v2(
        merged_prefix, tensor_names, shapes_and_slices, dtypes)
    ...
    # Restore the checkpoint values for the given Trackables.
  ```

  The restore function is executed after the non-registered restore ops.

  Args:
    package: Optional, the package that this class belongs to.
    name: (Required) The name of this saver, which is saved to the checkpoint.
      When a checkpoint is restored, the name and package are used to find the
      the matching restore function. The name and package are also used to
      generate a unique file prefix that is passed to the save_fn.
    predicate: (Required) A function that returns a boolean indicating whether a
      `Trackable` object should be checkpointed with this function. Predicates
      are executed in the reverse order that they are added (later registrations
      are checked first).
    save_fn: (Required) A function that takes a dictionary of trackables and a
      file prefix as the arguments, writes the checkpoint shards for the given
      Trackables, and returns the list of shard prefixes.
    restore_fn: (Required) A function that takes a dictionary of trackables and
      a file prefix as the arguments and restores the trackable values.
    strict_predicate_restore: If this is `True` (default), then an error will be
      raised if the predicate fails during checkpoint restoration. If this is
      `True`, checkpoint restoration will skip running the restore function.
      This value is generally set to `False` when the predicate does not pass on
      the Trackables after being saved/loaded from SavedModel.

  Raises:
    ValueError: if the package and name are already registered.
  """
    if not callable(save_fn):
        raise TypeError(f'The save_fn must be callable, got: {type(save_fn)}')
    if not callable(restore_fn):
        raise TypeError(f'The restore_fn must be callable, got: {type(restore_fn)}')
    _saver_registry.register(package, name, predicate, (save_fn, restore_fn, strict_predicate_restore))