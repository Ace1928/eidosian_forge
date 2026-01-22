from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def syslog(self, options, message):
    self.events.append((options, message))