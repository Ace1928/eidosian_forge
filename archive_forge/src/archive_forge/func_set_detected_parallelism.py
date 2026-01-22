from typing import Any, Dict, Optional, Union
from ray.data._internal.logical.operators.map_operator import AbstractMap
from ray.data.datasource.datasource import Datasource, Reader
def set_detected_parallelism(self, parallelism: int):
    """
        Set the true parallelism that should be used during execution. This
        should be specified by the user or detected by the optimizer.
        """
    self._detected_parallelism = parallelism