from zope.interface.verify import verifyObject
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, createMemoryWorker
def moreWork() -> None:
    done.append(7)