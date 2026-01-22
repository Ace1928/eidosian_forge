import itertools
from contextlib import ExitStack
def cleanUp(self):
    self._es.close()