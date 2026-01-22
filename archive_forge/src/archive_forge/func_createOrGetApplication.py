import getpass
import os
import pdb
import signal
import sys
import traceback
import warnings
from operator import attrgetter
from twisted import copyright, logger, plugin
from twisted.application import reactors, service
from twisted.application.reactors import NoSuchReactor, installReactor
from twisted.internet import defer
from twisted.internet.interfaces import _ISupportsExitSignalCapturing
from twisted.persisted import sob
from twisted.python import failure, log, logfile, runtime, usage, util
from twisted.python.reflect import namedAny, namedModule, qual
def createOrGetApplication(self):
    """
        Create or load an Application based on the parameters found in the
        given L{ServerOptions} instance.

        If a subcommand was used, the L{service.IServiceMaker} that it
        represents will be used to construct a service to be added to
        a newly-created Application.

        Otherwise, an application will be loaded based on parameters in
        the config.
        """
    if self.config.subCommand:
        plg = self.config.loadedPlugins[self.config.subCommand]
        ser = plg.makeService(self.config.subOptions)
        application = service.Application(plg.tapname)
        ser.setServiceParent(application)
    else:
        passphrase = getPassphrase(self.config['encrypted'])
        application = getApplication(self.config, passphrase)
    return application