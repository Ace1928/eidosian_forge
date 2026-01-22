from ase.io import write
import os
from ase.io.trajectory import Trajectory
from subprocess import Popen, PIPE
import time
def enough_jobs_running(self):
    """ Determines if sufficient jobs are running. """
    return self.number_of_jobs_running() >= self.n_simul