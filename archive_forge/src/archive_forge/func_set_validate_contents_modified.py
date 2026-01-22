from __future__ import annotations
import functools
import typing
import warnings
def set_validate_contents_modified(self, callback: Callable[[tuple[int, int, int], Collection[_T]], int | None]):
    """
        Assign a callback function to handle validating changes to the list.
        This may raise an exception if the change should not be performed.
        It may also return an integer position to be the new focus after the
        list is modified, or None to use the default behaviour.

        The callback is in the form:

        callback(indices, new_items)
        indices -- a (start, stop, step) tuple whose range covers the
            items being modified
        new_items -- an iterable of items replacing those at range(*indices),
            empty if items are being removed, if step==1 this list may
            contain any number of items
        """
    self._validate_contents_modified = callback