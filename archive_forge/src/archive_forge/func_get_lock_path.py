import copy
import os
from typing import Any, Dict
from ray._private.utils import get_ray_temp_dir
from ray.autoscaler._private.cli_logger import cli_logger
def get_lock_path(cluster_name: str) -> str:
    return os.path.join(get_ray_temp_dir(), 'cluster-{}.lock'.format(cluster_name))