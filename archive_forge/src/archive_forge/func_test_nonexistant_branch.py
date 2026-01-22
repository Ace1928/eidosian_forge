from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_nonexistant_branch(self):
    self.vfs_transport_factory = memory.MemoryServer
    location = self.get_url('absentdir/')
    out, err = self.run_bzr(['send', '--from', location], retcode=3)
    self.assertEqual(out, '')
    self.assertEqual(err, 'brz: ERROR: Not a branch: "%s".\n' % location)