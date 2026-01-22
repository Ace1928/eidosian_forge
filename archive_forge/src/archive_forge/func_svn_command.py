import pkg_resources
import sys
import optparse
from . import bool_optparse
import os
import re
import textwrap
from . import pluginlib
import configparser
import getpass
from logging.config import fileConfig
def svn_command(self, *args, **kw):
    """
        Run an svn command, but don't raise an exception if it fails.
        """
    try:
        return self.run_command('svn', *args, **kw)
    except OSError as e:
        if not self._svn_failed:
            print('Unable to run svn command (%s); proceeding anyway' % e)
            self._svn_failed = True