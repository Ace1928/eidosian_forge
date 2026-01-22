from unittest import TestResult
import testresources
from testresources.tests import TestUtil
def stopCleanResource(self, resource):
    self._calls.append(('clean', 'stop', resource))