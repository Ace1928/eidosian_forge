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
def test_files_have_timestamp(self):
    _dummy_timestamp = datetime.datetime(2013, 1, 1, 0, 0, 0, 0, UTC)
    self.patch(_o, 'create_timestamp', lambda: _dummy_timestamp)
    with temp_file_contents(b'Hello') as f:
        self.getUniqueString()
        result = get_result_for(['--attach-file', f.name])
        self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_bytes=b'Hello', timestamp=_dummy_timestamp), MatchesStatusCall(call='stopTestRun')]))