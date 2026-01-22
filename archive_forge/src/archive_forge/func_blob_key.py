from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@property
def blob_key(self):
    """Provide a key uniquely identifying a blob.

        Callers should consider these keys to be opaque-- i.e., to have
        no intrinsic meaning. Some data providers may use random IDs;
        but others may encode information into the key, in which case
        callers must make no attempt to decode it.
        """
    return self._blob_key