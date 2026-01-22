from __future__ import annotations
import functools
import re
import typing as ty
import warnings
def is_bad_version(self, version_str: str) -> bool:
    """Return True if `version_str` is too high

        Tests `version_str` with ``self.version_comparator``

        Parameters
        ----------
        version_str : str
            String giving version to test

        Returns
        -------
        is_bad : bool
            True if `version_str` is for version below that expected by
            ``self.version_comparator``, False otherwise.
        """
    return self.version_comparator(version_str) == -1