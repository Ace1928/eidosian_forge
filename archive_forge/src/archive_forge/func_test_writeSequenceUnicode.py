from unittest import skipIf
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_writeSequenceUnicode(self) -> None:
    """
        L{_pollingfile._PollableWritePipe.writeSequence} raises a C{TypeError}
        if unicode data is part of the data sequence to be appended to the
        output buffer.
        """
    p = _pollingfile._PollableWritePipe(1, lambda: None)
    self.assertRaises(TypeError, p.writeSequence, ['test'])
    self.assertRaises(TypeError, p.writeSequence, ('test',))