import copy
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from threading import Lock
from typing import Dict, List, Optional, Tuple
def batch_update(self, table: str, mutation: Dict[str, str]=None, deletion: List[str]=None, expected_version: Optional[int]=None) -> StoreStatus:
    mutation = mutation if mutation else {}
    deletion = deletion if deletion else []
    with self._lock:
        if expected_version is not None and expected_version != self._version:
            return StoreStatus(False, self._version)
        self._version += 1
        key_value_pairs_with_version = {key: VersionedValue(value, self._version) for key, value in mutation.items()}
        self._tables[table].update(key_value_pairs_with_version)
        for deleted_key in deletion:
            self._tables[table].pop(deleted_key, None)
        return StoreStatus(True, self._version)