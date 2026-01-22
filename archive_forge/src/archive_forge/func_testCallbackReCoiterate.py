from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def testCallbackReCoiterate(self):
    """
        If a callback to a deferred returned by coiterate calls coiterate on
        the same Cooperator, we should make sure to only do the minimal amount
        of scheduling work.  (This test was added to demonstrate a specific bug
        that was found while writing the scheduler.)
        """
    calls = []

    class FakeCall:

        def __init__(self, func):
            self.func = func

        def __repr__(self) -> str:
            return f'<FakeCall {self.func!r}>'

    def sched(f):
        self.assertFalse(calls, repr(calls))
        calls.append(FakeCall(f))
        return calls[-1]
    c = task.Cooperator(scheduler=sched, terminationPredicateFactory=lambda: lambda: True)
    d = c.coiterate(iter(()))
    done = []

    def anotherTask(ign):
        c.coiterate(iter(())).addBoth(done.append)
    d.addCallback(anotherTask)
    work = 0
    while not done:
        work += 1
        while calls:
            calls.pop(0).func()
            work += 1
        if work > 50:
            self.fail('Cooperator took too long')