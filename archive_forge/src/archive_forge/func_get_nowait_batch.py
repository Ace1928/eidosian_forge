import asyncio
from typing import Optional, Any, List, Dict
from collections.abc import Iterable
import ray
from ray.util.annotations import PublicAPI
def get_nowait_batch(self, num_items):
    if num_items > self.qsize():
        raise Empty(f'Cannot get {num_items} items from queue of size {self.qsize()}.')
    return [self.queue.get_nowait() for _ in range(num_items)]