import multiprocessing
import os
import re
import sys
import time
from .processes import ForkedProcess
from .remoteproxy import ClosedError
def runParallel(self):
    self.childs = []
    workers = self.workers
    chunks = [[] for i in range(workers)]
    i = 0
    for i in range(len(self.tasks)):
        chunks[i % workers].append(self.tasks[i])
    for i in range(workers):
        proc = ForkedProcess(target=None, preProxy=self.kwds, randomReseed=self.reseed)
        if not proc.isParent:
            self.proc = proc
            return Tasker(self, proc, chunks[i], proc.forkedProxies)
        else:
            self.childs.append(proc)
    self.progress = dict([(ch.childPid, []) for ch in self.childs])
    try:
        if self.showProgress:
            self.progressDlg.__enter__()
            self.progressDlg.setMaximum(len(self.tasks))
        activeChilds = self.childs[:]
        self.exitCodes = []
        pollInterval = 0.01
        while len(activeChilds) > 0:
            waitingChildren = 0
            rem = []
            for ch in activeChilds:
                try:
                    n = ch.processRequests()
                    if n > 0:
                        waitingChildren += 1
                except ClosedError:
                    rem.append(ch)
                    if self.showProgress:
                        self.progressDlg += 1
            for ch in rem:
                activeChilds.remove(ch)
                while True:
                    try:
                        pid, exitcode = os.waitpid(ch.childPid, 0)
                        self.exitCodes.append(exitcode)
                        break
                    except OSError as ex:
                        if ex.errno == 4:
                            continue
                        else:
                            raise
            if self.showProgress and self.progressDlg.wasCanceled():
                for ch in activeChilds:
                    ch.kill()
                raise CanceledError()
            if waitingChildren > 1:
                pollInterval *= 0.7
            elif waitingChildren == 0:
                pollInterval /= 0.7
            pollInterval = max(min(pollInterval, 0.5), 0.0005)
            time.sleep(pollInterval)
    finally:
        if self.showProgress:
            self.progressDlg.__exit__(None, None, None)
        for ch in self.childs:
            ch.join()
    if len(self.exitCodes) < len(self.childs):
        raise Exception('Parallelizer started %d processes but only received exit codes from %d.' % (len(self.childs), len(self.exitCodes)))
    for code in self.exitCodes:
        if code != 0:
            raise Exception('Error occurred in parallel-executed subprocess (console output may have more information).')
    return []