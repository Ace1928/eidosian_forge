import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_writing_not_completed(self):
    error = errors.WritingNotComplete('a request')
    self.assertEqualDiff("The MediumRequest 'a request' has not has finish_writing called upon it - until the write phase is complete no data may be read.", str(error))