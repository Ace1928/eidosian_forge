from os import getpid
from typing import Dict, List, Mapping, Optional, Sequence
from attrs import Factory, define
def inheritedNamedDescriptors(self) -> Dict[str, int]:
    """
        @return: A mapping from the names of configured descriptors to
            their integer values.
        """
    return dict(zip(self._names, self._descriptors))