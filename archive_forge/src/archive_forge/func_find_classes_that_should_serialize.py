import dataclasses
import inspect
import io
import pathlib
from dataclasses import dataclass
from typing import List, Type, Dict, Iterator, Tuple, Set
import numpy as np
import pandas as pd
import cirq
from cirq._import import ModuleType
from cirq.protocols.json_serialization import ObjectFactory
def find_classes_that_should_serialize(self) -> Set[Tuple[str, Type]]:
    result: Set[Tuple[str, Type]] = set()
    result.update({(name, obj) for name, obj in self._get_all_public_classes()})
    result.update(self.get_resolver_cache_types())
    return result