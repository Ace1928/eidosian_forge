import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, List, Optional
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def fallback_read_fn() -> Iterable[Block]:
    with _connect(self.connection_factory) as cursor:
        cursor.execute(self.sql)
        block = _cursor_to_block(cursor)
        return [block]