import os
import pickle
import sys
from zope.interface import Interface, implementer
from twisted.persisted import styles
from twisted.python import log, runtime
def guessType(filename):
    ext = os.path.splitext(filename)[1]
    return {'.tac': 'python', '.etac': 'python', '.py': 'python', '.tap': 'pickle', '.etap': 'pickle', '.tas': 'source', '.etas': 'source'}[ext]