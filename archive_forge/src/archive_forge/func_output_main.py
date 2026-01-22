import datetime
import sys
from functools import partial
from optparse import OptionGroup, OptionParser, OptionValueError
from subunit import make_stream_binary
from iso8601 import UTC
from subunit.v2 import StreamResultToBytes
def output_main():
    args = parse_arguments()
    output = StreamResultToBytes(sys.stdout)
    generate_stream_results(args, output)
    return 0