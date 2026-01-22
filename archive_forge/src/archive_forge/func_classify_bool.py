import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def classify_bool(seq: Iterable, pred: Callable) -> Any:
    false_elems = []
    true_elems = [elem for elem in seq if pred(elem) or false_elems.append(elem)]
    return (true_elems, false_elems)