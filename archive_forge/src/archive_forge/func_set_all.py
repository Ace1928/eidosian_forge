from collections import OrderedDict
from collections import abc
from typing import Any, Iterator, List, Tuple, Union
def set_all(self, key: MetadataKey, values: List[MetadataValue]) -> None:
    self._metadata[key] = values