import io
import os.path
from fixtures import TempDir
from testtools import TestCase
from testtools.matchers import FileContains
from subunit import _to_disk
from subunit.v2 import StreamResultToBytes
def test_smoke(self):
    output = os.path.join(self.useFixture(TempDir()).path, 'output')
    stdin = io.BytesIO()
    stdout = io.StringIO()
    writer = StreamResultToBytes(stdin)
    writer.startTestRun()
    writer.status('foo', 'success', {'tag'}, file_name='fred', file_bytes=b'abcdefg', eof=True, mime_type='text/plain')
    writer.stopTestRun()
    stdin.seek(0)
    _to_disk.to_disk(['-d', output], stdin=stdin, stdout=stdout)
    self.expectThat(os.path.join(output, 'foo/test.json'), FileContains('{"details": ["fred"], "id": "foo", "start": null, "status": "success", "stop": null, "tags": ["tag"]}'))
    self.expectThat(os.path.join(output, 'foo/fred'), FileContains('abcdefg'))