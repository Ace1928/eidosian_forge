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
def get_result_for(commands):
    """Get a result object from *commands.

    Runs the 'generate_stream_results' function from subunit._output after
    parsing *commands as if they were specified on the command line. The
    resulting bytestream is then converted back into a result object and
    returned.
    """
    result = StreamResult()
    args = safe_parse_arguments(commands)
    generate_stream_results(args, result)
    return result