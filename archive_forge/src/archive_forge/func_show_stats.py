import sys
from testtools import StreamToExtendedDecorator
from subunit import TestResultStats
from subunit.filters import run_filter_script
def show_stats(r):
    r.decorated.formatStats()