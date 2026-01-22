from __future__ import annotations
import re
from abc import abstractmethod
from collections import deque
from typing import AsyncIterator, Deque, Iterator, List, TypeVar, Union
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
def droplastn(iter: Iterator[T], n: int) -> Iterator[T]:
    """Drop the last n elements of an iterator."""
    buffer: Deque[T] = deque()
    for item in iter:
        buffer.append(item)
        if len(buffer) > n:
            yield buffer.popleft()