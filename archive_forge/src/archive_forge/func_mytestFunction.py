import os
import sys
from tempfile import mkdtemp
from shutil import rmtree
import pytest
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
def mytestFunction(insum=0):
    """
    Run a multiprocessing job and spawn child processes.
    """
    import multiprocessing
    import os
    import tempfile
    import time
    numberOfThreads = 2
    t = [None] * numberOfThreads
    a = [None] * numberOfThreads
    f = [None] * numberOfThreads

    def dummyFunction(filename):
        """
        This function writes the value 45 to the given filename.
        """
        j = 0
        for i in range(0, 10):
            j += i
        with open(filename, 'w') as f:
            f.write(str(j))
    for n in range(numberOfThreads):
        a[n] = True
        tmpFile = tempfile.mkstemp('.txt', 'test_engine_')[1]
        f[n] = tmpFile
        t[n] = multiprocessing.Process(target=dummyFunction, args=(tmpFile,))
        t[n].start()
    allDone = False
    while not allDone:
        time.sleep(1)
        for n in range(numberOfThreads):
            a[n] = t[n].is_alive()
        if not any(a):
            allDone = True
    total = insum
    for ff in f:
        with open(ff) as fd:
            total += int(fd.read())
        os.remove(ff)
    return total