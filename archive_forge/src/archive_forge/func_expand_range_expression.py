import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def expand_range_expression(range_exp):
    """Expand a range expression like '3-5' to values 3,4,5."""
    for part in range_exp.split(','):
        sub_range = part.split('-')
        if len(sub_range) == 1:
            sub_range = sub_range * 2
        else:
            assert len(sub_range) == 2
        num_digits = len(sub_range[0])
        for i in range(int(sub_range[0]), int(sub_range[1]) + 1):
            yield str(i).zfill(num_digits)