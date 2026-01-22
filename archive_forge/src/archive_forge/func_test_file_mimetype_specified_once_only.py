import datetime
import optparse
from contextlib import contextmanager
from functools import partial
from io import BytesIO, TextIOWrapper
from tempfile import NamedTemporaryFile
from iso8601 import UTC
from testtools import TestCase
from testtools.matchers import (Equals, Matcher, MatchesAny, MatchesListwise,
from testtools.testresult.doubles import StreamResult
import subunit._output as _o
from subunit._output import (_ALL_ACTIONS, _FINAL_ACTIONS,
def test_file_mimetype_specified_once_only(self):
    with temp_file_contents(b'Hi') as f:
        self.patch(_o, '_CHUNK_SIZE', 1)
        result = get_result_for([self.option, self.test_id, '--attach-file', f.name, '--mimetype', 'text/plain'])
        self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_id=self.test_id, mime_type='text/plain', file_bytes=b'H', eof=False), MatchesStatusCall(test_id=self.test_id, mime_type=None, file_bytes=b'i', eof=True), MatchesStatusCall(call='stopTestRun')]))