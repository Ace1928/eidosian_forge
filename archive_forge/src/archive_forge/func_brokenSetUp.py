from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
def brokenSetUp(self):
    self.log = ['setUp']
    raise RuntimeError('Deliberate failure')