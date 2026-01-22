from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'Use `tf.data.Dataset.choose_from_datasets(...)` instead. Note that, unlike the experimental endpoint, the non-experimental endpoint sets `stop_on_empty_dataset=True` by default. You should set this argument explicitly in case you would like to match the behavior of the experimental endpoint.')
@tf_export(v1=['data.experimental.choose_from_datasets'])
def choose_from_datasets_v1(datasets, choice_dataset, stop_on_empty_dataset=False):
    return dataset_ops.DatasetV1Adapter(choose_from_datasets_v2(datasets, choice_dataset, stop_on_empty_dataset))