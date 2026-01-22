from __future__ import annotations
import io
import typing as ty
from copy import copy
from .openers import ImageOpener
def same_file_as(self, other: FileHolder) -> bool:
    """Test if `self` refers to same files / fileobj as `other`

        Parameters
        ----------
        other : object
            object with `filename` and `fileobj` attributes

        Returns
        -------
        tf : bool
            True if `other` has the same filename (or both have None) and the
            same fileobj (or both have None
        """
    return self.filename == other.filename and self.fileobj == other.fileobj