import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
Read a list of lines from the log file.

        This doesn't returns all of the files lines - call it multiple times.
        