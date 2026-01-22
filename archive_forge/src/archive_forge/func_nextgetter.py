import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def nextgetter(obj):
    return obj.next