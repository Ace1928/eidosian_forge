import re
import sys
from optparse import OptionParser
from testtools import ExtendedToStreamDecorator, StreamToExtendedDecorator
from subunit import StreamResultToBytes, read_test_list
from subunit.filters import filter_by_result, find_stream
from subunit.test_results import (TestResultFilter, and_predicates,
def only_genuine_failures_callback(option, opt, value, parser):
    parser.rargs.insert(0, '--no-passthrough')
    parser.rargs.insert(0, '--no-xfail')
    parser.rargs.insert(0, '--no-skip')
    parser.rargs.insert(0, '--no-success')