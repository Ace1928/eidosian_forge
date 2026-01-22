from ase.io import write
import os
from ase.io.trajectory import Trajectory
from subprocess import Popen, PIPE
import time
def number_of_jobs_running(self):
    """ Determines how many jobs are running. The user
            should use this or the enough_jobs_running method
            to verify that a job needs to be started before
            calling the relax method."""
    self.__cleanup__()
    p = Popen(['`which {0}` -u `whoami`'.format(self.qstat_command)], shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True, universal_newlines=True)
    fout = p.stdout
    lines = fout.readlines()
    n_running = 0
    for l in lines:
        if l.find(self.job_prefix) != -1:
            n_running += 1
    return n_running