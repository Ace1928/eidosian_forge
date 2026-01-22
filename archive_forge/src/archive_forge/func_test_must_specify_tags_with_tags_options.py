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
def test_must_specify_tags_with_tags_options(self):

    def fn():
        return safe_parse_arguments(['--fail', 'foo', '--tag'])
    self.assertThat(fn, MatchesAny(raises(RuntimeError('--tag option requires 1 argument')), raises(RuntimeError('--tag option requires an argument'))))