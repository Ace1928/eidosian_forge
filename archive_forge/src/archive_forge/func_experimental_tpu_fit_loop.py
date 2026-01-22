import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils_v1 as dist_utils
from tensorflow.python.keras.engine import partial_batch_padding_handler as padding_util
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
def experimental_tpu_fit_loop(model, dataset, epochs=100, verbose=1, callbacks=None, initial_epoch=0, steps_per_epoch=None, val_dataset=None, validation_steps=None, validation_freq=1):
    """Fit loop for training with TPU tf.distribute.Strategy.

  Args:
      model: Keras Model instance.
      dataset: Dataset that returns inputs and targets
      epochs: Number of times to iterate over the data
      verbose: Integer, Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      initial_epoch: Epoch at which to start training
          (useful for resuming a previous training run)
      steps_per_epoch: Total number of steps (batches of samples)
          before declaring one epoch finished and starting the
          next epoch. Ignored with the default value of `None`.
      val_dataset: Dataset for validation data.
      validation_steps: Number of steps to run validation for
          (only if doing validation from data tensors).
          Ignored with the default value of `None`.
      validation_freq: Only relevant if validation data is provided. Integer or
          `collections.abc.Container` instance (e.g. list, tuple, etc.). If an
          integer, specifies how many training epochs to run before a new
          validation run is performed, e.g. `validation_freq=2` runs
          validation every 2 epochs. If a Container, specifies the epochs on
          which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
          validation at the end of the 1st, 2nd, and 10th epochs.

  Returns:
      Returns `None`.

  Raises:
      ValueError: in case of invalid arguments.
  """
    mode = ModeKeys.TRAIN
    current_strategy = model._distribution_strategy
    iteration_value = min(steps_per_epoch, current_strategy.extended.steps_per_run)
    steps_per_run = backend.variable(value=iteration_value, dtype='int32', name='steps_per_run')
    iterator = dist_utils.get_iterator(dataset, current_strategy)
    scope = dist_utils.distributed_scope(strategy=current_strategy, learning_phase=1)
    scope.__enter__()
    out_labels = model.metrics_names or []
    step_fn = _make_train_step_fn(model, ModeKeys.TRAIN, current_strategy, out_labels)
    initial_loop_values = {}
    initial_loop_values['loss'] = constant_op.constant(10000000.0)
    for m in model._get_training_eval_metrics():
        tensor = m.result()
        initial_loop_values[m.name] = array_ops.zeros(tensor.shape, tensor.dtype)
    ctx = current_strategy.extended.experimental_run_steps_on_iterator(step_fn, iterator, iterations=steps_per_run, initial_loop_values=initial_loop_values)
    train_op = ctx.run_op
    output_tensors = ctx.last_step_outputs
    do_validation = bool(validation_steps)
    if model._compile_distribution:
        dist_utils._copy_weights_to_distributed_model(model, mode)
    callbacks = cbks.configure_callbacks(callbacks, model, do_validation=do_validation, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=verbose, count_mode='steps', mode=mode)
    steps_to_run = [current_strategy.extended.steps_per_run] * (steps_per_epoch // current_strategy.extended.steps_per_run)
    if steps_per_epoch % current_strategy.extended.steps_per_run:
        steps_to_run.append(steps_per_epoch % current_strategy.extended.steps_per_run)
    target_steps = len(steps_to_run)
    callbacks._call_begin_hook(mode)
    initial_epoch = model._maybe_load_initial_epoch_from_ckpt(initial_epoch, mode)
    for epoch in range(initial_epoch, epochs):
        dist_utils._reset_metrics(model)
        callbacks.on_epoch_begin(epoch)
        epoch_logs = {}
        step_index = 0
        prev_step_count = None
        current_step = 0
        while current_step < target_steps:
            step_count = steps_to_run[current_step]
            batch_logs = {'batch': step_index, 'size': 1, 'num_steps': step_count}
            callbacks._call_batch_hook(mode, 'begin', step_index, batch_logs)
            if prev_step_count is None or step_count != prev_step_count:
                backend.get_session().run(steps_per_run.assign(step_count))
                prev_step_count = step_count
            try:
                _, outputs = backend.batch_get_value([train_op, output_tensors])
            except errors.OutOfRangeError:
                logging.warning('Your dataset iterator ran out of data; interrupting training. Make sure that your dataset can generate at least `steps_per_epoch * epochs` batches (in this case, %d batches).' % steps_per_epoch * epochs)
                break
            batch_logs.update(outputs)
            callbacks._call_batch_hook(mode, 'end', step_index, batch_logs)
            step_index = step_index + step_count
            current_step += 1
            if callbacks.model.stop_training:
                break
        if do_validation and training_utils_v1.should_run_validation(validation_freq, epoch):
            logging.info('Running validation at fit epoch: %s', epoch)
            if model._compile_distribution:
                dist_utils._copy_weights_to_original_model(model, ModeKeys.TRAIN)
            val_outs = experimental_tpu_test_loop(model, val_dataset, steps=validation_steps, verbose=verbose, callbacks=callbacks)
            if not isinstance(val_outs, list):
                val_outs = [val_outs]
            for label, val_out in zip(out_labels, val_outs):
                epoch_logs['val_' + label] = val_out
        callbacks.on_epoch_end(epoch, epoch_logs)
        if callbacks.model.stop_training:
            break
    model._successful_loop_finish = True
    callbacks._call_end_hook(mode)
    if model._compile_distribution:
        dist_utils._copy_weights_to_original_model(model, ModeKeys.TRAIN)
    scope.__exit__(None, None, None)
    return model.history