from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def createWorker():
    if self.noMoreWorkers():
        return None
    worker, performer = createMemoryWorker()
    self.workerPerformers.append(performer)
    self.activePerformers.append(performer)
    cw = ContextualWorker(worker, worker=len(self.workerPerformers))
    self.allWorkersEver.append(cw)
    self.allUnquitWorkers.append(cw)
    realQuit = cw.quit

    def quitAndRemove():
        realQuit()
        self.allUnquitWorkers.remove(cw)
        self.activePerformers.remove(performer)
    cw.quit = quitAndRemove
    return cw