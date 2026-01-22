from abc import ABC, abstractmethod
from .header import Field
@property
def tractogram(self):
    return self._tractogram