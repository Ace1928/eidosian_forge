import multiprocessing
import os
import re
import sys
import time
from .processes import ForkedProcess
from .remoteproxy import ClosedError
def numWorkers(self):
    """
        Return the number of parallel workers
        """
    return self.par.workers