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
def test_can_parse_all_commands_with_test_id(self):
    test_id = self.getUniqueString()
    args = safe_parse_arguments(args=[self.option, test_id])
    self.assertThat(args.action, Equals(self.command))
    self.assertThat(args.test_id, Equals(test_id))