from unittest import TestResult
import testresources
from testresources.tests import TestUtil
def stopMakeResource(self, resource):
    self._calls.append(('make', 'stop', resource))