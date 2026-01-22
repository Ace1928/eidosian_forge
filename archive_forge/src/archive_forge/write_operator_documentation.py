from typing import Any, Dict, Optional, Union
from ray.data._internal.compute import TaskPoolStrategy
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.map_operator import AbstractMap
from ray.data.datasource.datasink import Datasink
from ray.data.datasource.datasource import Datasource
Logical operator for write.