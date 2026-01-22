from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
def has_subkey(self, subkey: str | AdfKey) -> bool:
    """
        Return True if this AdfKey contains the given subkey.

        Args:
            subkey (str or AdfKey): A key name or an AdfKey object.

        Returns:
            bool: Whether this key contains the given key.
        """
    if isinstance(subkey, str):
        key = subkey
    elif isinstance(subkey, AdfKey):
        key = subkey.key
    else:
        raise ValueError('The subkey should be an AdfKey or a string!')
    if len(self.subkeys) > 0 and key in (k.key for k in self.subkeys):
        return True
    return False