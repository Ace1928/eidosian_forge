from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys as _feature_keys
from tensorflow_estimator.python.estimator.canned.timeseries import head as _head
from tensorflow_estimator.python.estimator.canned.timeseries import model_utils as _model_utils
def cold_start_filter(signatures, session, features):
    """Perform filtering using an exported saved model.

  Filtering refers to updating model state based on new observations.
  Predictions based on the returned model state will be conditioned on these
  observations.

  Starts from the model's default/uninformed state.

  Args:
    signatures: The `MetaGraphDef` protocol buffer returned from
      `tf.saved_model.loader.load`. Used to determine the names of Tensors to
      feed and fetch. Must be from the same model as `continue_from`.
    session: The session to use. The session's graph must be the one into which
      `tf.saved_model.loader.load` loaded the model.
    features: A dictionary mapping keys to Numpy arrays, with several possible
      shapes (requires keys `FilteringFeatures.TIMES` and
      `FilteringFeatures.VALUES`): Single example; `TIMES` is a scalar and
        `VALUES` is either a scalar or a vector of length [number of features].
        Sequence; `TIMES` is a vector of shape [series length], `VALUES` either
        has shape [series length] (univariate) or [series length x number of
        features] (multivariate). Batch of sequences; `TIMES` is a vector of
        shape [batch size x series length], `VALUES` has shape [batch size x
        series length] or [batch size x series length x number of features]. In
        any case, `VALUES` and any exogenous features must have their shapes
        prefixed by the shape of the value corresponding to the `TIMES` key.

  Returns:
    A dictionary containing model state updated to account for the observations
    in `features`.
  """
    filter_signature = signatures.signature_def[_feature_keys.SavedModelLabels.COLD_START_FILTER]
    features = _canonicalize_numpy_data(data=features, require_single_batch=False)
    output_tensors_by_name, feed_dict = _colate_features_to_feeds_and_fetches(signature=filter_signature, features=features, graph=session.graph)
    output = session.run(output_tensors_by_name, feed_dict=feed_dict)
    output[_feature_keys.FilteringResults.TIMES] = features[_feature_keys.FilteringFeatures.TIMES]
    return output