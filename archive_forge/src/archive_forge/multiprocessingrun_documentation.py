from multiprocessing import Pool
import time
from ase.io import write, read
Checks that all calculations are finished, if not
        wait and check again. Return when all are finished.