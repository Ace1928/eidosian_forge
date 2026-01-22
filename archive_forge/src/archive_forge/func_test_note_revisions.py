from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_note_revisions(self):
    stderr = self.run_send([])[1]
    self.assertEndsWith(stderr, '\nBundling 1 revision.\n')