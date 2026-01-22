import io
import os
import os.path
import sys
import warnings
import cherrypy
def statfiles(self):
    """:rtype: list of available profiles.
        """
    return [f for f in os.listdir(self.path) if f.startswith('cp_') and f.endswith('.prof')]