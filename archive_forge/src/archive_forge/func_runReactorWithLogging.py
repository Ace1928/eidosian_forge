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
def runReactorWithLogging(config, oldstdout, oldstderr, profiler=None, reactor=None):
    """
    Start the reactor, using profiling if specified by the configuration, and
    log any error happening in the process.

    @param config: configuration of the twistd application.
    @type config: L{ServerOptions}

    @param oldstdout: initial value of C{sys.stdout}.
    @type oldstdout: C{file}

    @param oldstderr: initial value of C{sys.stderr}.
    @type oldstderr: C{file}

    @param profiler: object used to run the reactor with profiling.
    @type profiler: L{AppProfiler}

    @param reactor: The reactor to use.  If L{None}, the global reactor will
        be used.
    """
    if reactor is None:
        from twisted.internet import reactor
    try:
        if config['profile']:
            if profiler is not None:
                profiler.run(reactor)
        elif config['debug']:
            sys.stdout = oldstdout
            sys.stderr = oldstderr
            if runtime.platformType == 'posix':
                signal.signal(signal.SIGUSR2, lambda *args: pdb.set_trace())
                signal.signal(signal.SIGINT, lambda *args: pdb.set_trace())
            fixPdb()
            pdb.runcall(reactor.run)
        else:
            reactor.run()
    except BaseException:
        close = False
        if config['nodaemon']:
            file = oldstdout
        else:
            file = open('TWISTD-CRASH.log', 'a')
            close = True
        try:
            traceback.print_exc(file=file)
            file.flush()
        finally:
            if close:
                file.close()