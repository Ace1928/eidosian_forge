import glob
import operator
import os
import shutil
import sys
import tempfile
from incremental import Version
from twisted.python import release
from twisted.python._release import (
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def commitRepository(self, repository):
    """
        Add and commit all the files from the Git repository specified.

        @type repository: L{FilePath}
        @params repository: The Git repository to commit into.
        """
    runCommand(['git', '-C', repository.path, 'add'] + glob.glob(repository.path + '/*'))
    runCommand(['git', '-C', repository.path, 'commit', '-m', 'hop'])