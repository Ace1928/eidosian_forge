import sys
from optparse import OptionParser
from testtools import ExtendedToStreamDecorator
from subunit import StreamResultToBytes
from subunit.filters import find_stream, run_tests_from_stream
Convert a version 1 subunit stream to version 2 stream.