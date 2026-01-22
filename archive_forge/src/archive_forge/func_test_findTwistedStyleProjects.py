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
def test_findTwistedStyleProjects(self):
    """
        findTwistedStyleProjects finds all projects underneath a particular
        directory. A 'project' is defined by the existence of a 'newsfragments'
        directory and is returned as a Project object.
        """
    baseDirectory = self.makeProjects(('foo', 2, 3, 0), ('foo.bar', 0, 7, 4))
    projects = findTwistedProjects(baseDirectory)
    self.assertProjectsEqual(projects, [Project(baseDirectory.child('foo')), Project(baseDirectory.child('foo').child('bar'))])