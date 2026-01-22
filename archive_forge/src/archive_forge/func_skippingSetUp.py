from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
def skippingSetUp(self):
    self.log = ['setUp']
    raise SkipTest("Don't do this")