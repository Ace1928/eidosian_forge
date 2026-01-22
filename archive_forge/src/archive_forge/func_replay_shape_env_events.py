import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def replay_shape_env_events(events):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    constructor_event = events[0]
    assert constructor_event.f == ShapeEnv
    shape_env = constructor_event.run()
    for event in events[1:]:
        try:
            event.run(shape_env)
        except Exception as e:
            raise RuntimeError(f'failed when running event: {event}') from e
    return shape_env