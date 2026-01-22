import itertools
from typing import TYPE_CHECKING, Type, Callable, Dict, Optional, Union, Iterable, Sequence, List
from cirq import ops, circuits, protocols, _import
from cirq.transformers import transformer_api
Get the "class" of an operator, by index.