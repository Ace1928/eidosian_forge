from unittest import TestResult
import testresources
from testresources.tests import TestUtil
def startCleanResource(self, resource):
    self._calls.append(('clean', 'start', resource))