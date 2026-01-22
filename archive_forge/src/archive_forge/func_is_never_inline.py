from abc import ABCMeta, abstractmethod
@property
def is_never_inline(self):
    """
        True if never inline
        """
    return self._inline == 'never'