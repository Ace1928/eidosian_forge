import parlai.utils.logging as logging
from abc import ABC, abstractmethod
from typing import Dict
@staticmethod
@abstractmethod
def default_maxlen():
    raise NotImplemented