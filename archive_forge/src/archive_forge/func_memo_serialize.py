import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def memo_serialize(self, types_to_memoize: List) -> Any:
    memo = SerializeMemoizer(types_to_memoize)
    return (self.serialize(memo), memo.serialize())