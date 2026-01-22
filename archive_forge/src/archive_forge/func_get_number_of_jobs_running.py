from subprocess import Popen, PIPE
import os
import time
from ase.io import write, read
def get_number_of_jobs_running(self):
    """ Returns the number of jobs running.
             It is a good idea to check that this is 0 before
             terminating the main program. """
    self.__cleanup__()
    return len(self.running_pids)