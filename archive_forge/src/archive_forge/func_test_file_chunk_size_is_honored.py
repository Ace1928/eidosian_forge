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
def test_file_chunk_size_is_honored(self):
    with temp_file_contents(b'Hello') as f:
        self.patch(_o, '_CHUNK_SIZE', 1)
        result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
        self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_id=self.test_id, file_bytes=b'H', eof=False), MatchesStatusCall(test_id=self.test_id, file_bytes=b'e', eof=False), MatchesStatusCall(test_id=self.test_id, file_bytes=b'l', eof=False), MatchesStatusCall(test_id=self.test_id, file_bytes=b'l', eof=False), MatchesStatusCall(test_id=self.test_id, file_bytes=b'o', eof=True), MatchesStatusCall(call='stopTestRun')]))