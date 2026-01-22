import re
import itertools
def dataPosIterator(self):
    if not self._dataPosList:
        self._dataPosList = list(self._dataPosIterator())
    return self._dataPosList