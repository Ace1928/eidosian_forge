import tempfile
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
def checkCoordinatedReadGroups(self, results, num_consumers):
    """Validates results from a `make_coordinted_read_dataset` dataset.

    Each group of `num_consumers` results should be consecutive, indicating that
    they were produced by the same worker.

    Args:
      results: The elements produced by the dataset.
      num_consumers: The number of consumers.
    """
    groups = [results[start:start + num_consumers] for start in range(0, len(results), num_consumers)]
    incorrect_groups = []
    for group in groups:
        for offset in range(1, len(group)):
            if group[0] + offset != group[offset]:
                incorrect_groups.append(group)
                break
    self.assertEmpty(incorrect_groups, 'Incorrect groups: {}.\nAll groups: {}'.format(incorrect_groups, groups))