import threading
import time
from abc import ABCMeta, abstractmethod
def proof(self, simplify=True):
    """
        Return the proof string
        :param simplify: bool simplify the proof?
        :return: str
        """
    if self._result is None:
        raise LookupError('You have to call prove() first to get a proof!')
    else:
        return self.decorate_proof(self._proof, simplify)