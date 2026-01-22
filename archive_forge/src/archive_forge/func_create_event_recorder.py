from collections import defaultdict, deque
from functools import partial
import statistics
from typing import ClassVar, Deque, Dict, Optional
import torch
def create_event_recorder(event_name: str, dummy: bool=False) -> EventRecorder:
    if not dummy:
        return CudaEventRecorder(event_name)
    return DummyCudaEventRecorder()