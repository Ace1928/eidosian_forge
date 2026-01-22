from collections import OrderedDict
from collections import abc
from typing import Any, Iterator, List, Tuple, Union
def delete_all(self, key: MetadataKey) -> None:
    """Delete all mappings for <key>."""
    del self._metadata[key]