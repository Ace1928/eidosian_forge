import gi
import sys
from gi.repository import Notify  # noqa: E402
from testtools import StreamToExtendedDecorator  # noqa: E402
from subunit import TestResultStats  # noqa: E402
from subunit.filters import run_filter_script  # noqa: E402
Notify the user of a finished test run.