import sys
from optparse import OptionParser
from testtools import StreamResultRouter, StreamToExtendedDecorator
from subunit import ByteStreamToStreamResult, TestProtocolClient
from subunit.filters import find_stream
from subunit.test_results import CatFiles
Convert a version 2 subunit stream to a version 1 stream.