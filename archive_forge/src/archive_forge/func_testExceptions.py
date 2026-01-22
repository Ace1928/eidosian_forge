from twisted.python import roots
from twisted.trial import unittest
def testExceptions(self) -> None:
    request = roots.Request()
    try:
        request.write(b'blah')
    except NotImplementedError:
        pass
    else:
        self.fail()
    try:
        request.finish()
    except NotImplementedError:
        pass
    else:
        self.fail()