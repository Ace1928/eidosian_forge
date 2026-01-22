import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def progress_str(self) -> str:
    base = f'{self._actor_pool.num_running_actors()} actors'
    pending = self._actor_pool.num_pending_actors()
    if pending:
        base += f' ({pending} pending)'
    if self._actor_locality_enabled:
        base += ' ' + locality_string(self._actor_pool._locality_hits, self._actor_pool._locality_misses)
    else:
        base += ' [locality off]'
    return base