from __future__ import print_function
import os,stat,time
import errno
import sys
def getlink(self, folder):
    """ Get a convenient link for accessing items  """
    return PickleShareLink(self, folder)