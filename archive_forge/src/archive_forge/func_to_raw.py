import os
import pathlib
import tempfile
import uuid
import numpy as np
from ..utils import is_soundfile_availble, is_torch_available, is_vision_available, logging
def to_raw(self):
    """
        Returns the "raw" version of that object. It is a `torch.Tensor` object.
        """
    if self._tensor is not None:
        return self._tensor
    if self._path is not None:
        tensor, self.samplerate = sf.read(self._path)
        self._tensor = torch.tensor(tensor)
        return self._tensor