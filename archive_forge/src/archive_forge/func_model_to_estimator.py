from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
from absl import logging
import tensorflow as tf
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def model_to_estimator(keras_model=None, keras_model_path=None, custom_objects=None, model_dir=None, config=None, checkpoint_format=None, use_v2_estimator=False, metric_names_map=None, export_outputs=None):
    """Constructs an `Estimator` instance from given keras model.

  If you use infrastructure or other tooling that relies on Estimators, you can
  still build a Keras model and use model_to_estimator to convert the Keras
  model to an Estimator for use with downstream systems.

  For usage example, please see:
  [Creating estimators from Keras
  Models](https://www.tensorflow.org/guide/estimator#create_an_estimator_from_a_keras_model).

  Sample Weights:
  Estimators returned by `model_to_estimator` are configured so that they can
  handle sample weights (similar to `keras_model.fit(x, y, sample_weights)`).

  To pass sample weights when training or evaluating the Estimator, the first
  item returned by the input function should be a dictionary with keys
  `features` and `sample_weights`. Example below:

  ```python
  keras_model = tf.keras.Model(...)
  keras_model.compile(...)

  estimator = tf.keras.estimator.model_to_estimator(keras_model)

  def input_fn():
    return dataset_ops.Dataset.from_tensors(
        ({'features': features, 'sample_weights': sample_weights},
         targets))

  estimator.train(input_fn, steps=1)
  ```

  Example with customized export signature:
  ```python
  inputs = {'a': tf.keras.Input(..., name='a'),
            'b': tf.keras.Input(..., name='b')}
  outputs = {'c': tf.keras.layers.Dense(..., name='c')(inputs['a']),
             'd': tf.keras.layers.Dense(..., name='d')(inputs['b'])}
  keras_model = tf.keras.Model(inputs, outputs)
  keras_model.compile(...)
  export_outputs = {'c': tf.estimator.export.RegressionOutput,
                    'd': tf.estimator.export.ClassificationOutput}

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model, export_outputs=export_outputs)

  def input_fn():
    return dataset_ops.Dataset.from_tensors(
        ({'features': features, 'sample_weights': sample_weights},
         targets))

  estimator.train(input_fn, steps=1)
  ```

  Note: We do not support creating weighted metrics in Keras and converting them
  to weighted metrics in the Estimator API using `model_to_estimator`.
  You will have to create these metrics directly on the estimator spec using the
  `add_metrics` function.

  Args:
    keras_model: A compiled Keras model object. This argument is mutually
      exclusive with `keras_model_path`. Estimator's `model_fn` uses the
      structure of the model to clone the model. Defaults to `None`.
    keras_model_path: Path to a compiled Keras model saved on disk, in HDF5
      format, which can be generated with the `save()` method of a Keras model.
      This argument is mutually exclusive with `keras_model`.
      Defaults to `None`.
    custom_objects: Dictionary for cloning customized objects. This is
      used with classes that is not part of this pip package. For example, if
      user maintains a `relu6` class that inherits from `tf.keras.layers.Layer`,
      then pass `custom_objects={'relu6': relu6}`. Defaults to `None`.
    model_dir: Directory to save `Estimator` model parameters, graph, summary
      files for TensorBoard, etc. If unset a directory will be created with
      `tempfile.mkdtemp`
    config: `RunConfig` to config `Estimator`. Allows setting up things in
      `model_fn` based on configuration such as `num_ps_replicas`, or
      `model_dir`. Defaults to `None`. If both `config.model_dir` and the
      `model_dir` argument (above) are specified the `model_dir` **argument**
      takes precedence.
    checkpoint_format: Sets the format of the checkpoint saved by the estimator
      when training. May be `saver` or `checkpoint`, depending on whether to
      save checkpoints from `tf.compat.v1.train.Saver` or `tf.train.Checkpoint`.
      The default is `checkpoint`. Estimators use name-based `tf.train.Saver`
      checkpoints, while Keras models use object-based checkpoints from
      `tf.train.Checkpoint`. Currently, saving object-based checkpoints from
      `model_to_estimator` is only supported by Functional and Sequential
      models.
    use_v2_estimator: Whether to convert the model to a V2 Estimator or V1
      Estimator. Defaults to `False`.
    metric_names_map: Optional dictionary mapping Keras model output metric
      names to custom names. This can be used to override the default Keras
      model output metrics names in a multi IO model use case and provide custom
      names for the `eval_metric_ops` in Estimator.
      The Keras model metric names can be obtained using `model.metrics_names`
      excluding any loss metrics such as total loss and output losses.
      For example, if your Keras model has two outputs `out_1` and `out_2`,
      with `mse` loss and `acc` metric, then `model.metrics_names` will be
      `['loss', 'out_1_loss', 'out_2_loss', 'out_1_acc', 'out_2_acc']`.
      The model metric names excluding the loss metrics will be
      `['out_1_acc', 'out_2_acc']`.
    export_outputs: Optional dictionary. This can be used to override the
      default Keras model output exports in a multi IO model use case and
      provide custom names for the `export_outputs` in
      `tf.estimator.EstimatorSpec`. Default is None, which is equivalent to
      {'serving_default': `tf.estimator.export.PredictOutput`}.
      A dict `{name: output}` where:
        * name: An arbitrary name for this output. This becomes the signature
          name in the SavedModel.
        * output: an `ExportOutput` object such as `ClassificationOutput`,
          `RegressionOutput`, or `PredictOutput`. Single-headed models only need
          to specify one entry in this dictionary. Multi-headed models should
          specify one entry for each head, one of which must be named using
          `tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`.
          If no entry is provided, a default `PredictOutput` mapping to
          `predictions` will be created.

  Returns:
    An Estimator from given keras model.

  Raises:
    ValueError: If neither keras_model nor keras_model_path was given.
    ValueError: If both keras_model and keras_model_path was given.
    ValueError: If the keras_model_path is a GCS URI.
    ValueError: If keras_model has not been compiled.
    ValueError: If an invalid checkpoint_format was given.
  """
    if not (keras_model or keras_model_path):
        raise ValueError('Either `keras_model` or `keras_model_path` needs to be provided.')
    if keras_model and keras_model_path:
        raise ValueError('Please specity either `keras_model` or `keras_model_path`, but not both.')
    if keras_model:
        _assert_valid_model(keras_model, custom_objects)
    config = estimator_lib.maybe_overwrite_model_dir_and_session_config(config, model_dir)
    if not keras_model:
        if keras_model_path.startswith('gs://') or 'storage.googleapis.com' in keras_model_path:
            keras_model_path = _get_file_from_google_storage(keras_model_path, config.model_dir)
        tf.compat.v1.logging.info('Loading models from %s', keras_model_path)
        keras_model = tf.keras.models.load_model(keras_model_path)
    else:
        tf.compat.v1.logging.info('Using the Keras model provided.')
        keras_model = keras_model
    if checkpoint_format is None or checkpoint_format == 'checkpoint':
        if not (keras_model._is_graph_network or isinstance(keras_model, tf.keras.models.Sequential)):
            raise ValueError('Object-based checkpoints are currently not supported with subclassed models.')
        save_object_ckpt = True
    elif checkpoint_format == 'saver':
        save_object_ckpt = False
    else:
        raise ValueError('Checkpoint format must be one of "checkpoint" or "saver". Got {}'.format(checkpoint_format))
    if not hasattr(keras_model, 'optimizer') or not keras_model.optimizer:
        raise ValueError('The given keras model has not been compiled yet. Please compile the model with `model.compile()` before calling `model_to_estimator()`.')
    keras_model_fn = _create_keras_model_fn(keras_model, custom_objects, save_object_ckpt, metric_names_map, export_outputs)
    if _any_weight_initialized(keras_model):
        if config.session_config.HasField('gpu_options'):
            tf.compat.v1.logging.warn('The Keras backend session has already been set. The _session_config passed to model_to_estimator will not be used.')
    else:
        sess = tf.compat.v1.Session(config=config.session_config)
        tf.compat.v1.keras.backend.set_session(sess)
    warm_start_path = None
    if keras_model._is_graph_network and config.is_chief:
        warm_start_path = _save_first_checkpoint(keras_model, custom_objects, config, save_object_ckpt)
    elif keras_model.built:
        tf.compat.v1.logging.warn("You are creating an Estimator from a Keras model manually subclassed from `Model`, that was already called on some inputs (and thus already had weights). We are currently unable to preserve the model's state (its weights) as part of the estimator in this case. Be warned that the estimator has been created using a freshly initialized version of your model.\nNote that this doesn't affect the state of the model instance you passed as `keras_model` argument.")
    if use_v2_estimator:
        estimator_cls = estimator_lib.EstimatorV2
    else:
        estimator_cls = estimator_lib.Estimator
    estimator = estimator_cls(keras_model_fn, config=config, warm_start_from=warm_start_path)
    return estimator