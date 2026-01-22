from twisted.trial.unittest import SynchronousTestCase
from .._convenience import Quit
from .._ithreads import AlreadyQuit

        L{Quit.set} raises L{AlreadyQuit} if it has been called previously.
        