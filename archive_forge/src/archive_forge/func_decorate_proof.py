import threading
import time
from abc import ABCMeta, abstractmethod
def decorate_proof(self, proof_string, simplify=True):
    """
        Modify and return the proof string
        :param proof_string: str the proof to decorate
        :param simplify: bool simplify the proof?
        :return: str
        """
    return self._command.decorate_proof(proof_string, simplify)