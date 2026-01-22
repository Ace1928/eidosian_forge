import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
def binary_locations(self):
    """
        A list of directories that should be searched for the prover9
        executables.  This list is used by ``config_prover9`` when searching
        for the prover9 executables.
        """
    return ['/usr/local/bin/prover9', '/usr/local/bin/prover9/bin', '/usr/local/bin', '/usr/bin', '/usr/local/prover9', '/usr/local/share/prover9']