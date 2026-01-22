import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO

        :param i_line:
        :param instruction:
        :param tok:
        :param priority:
        :param after:
        :param end_of_line:
            Marker to signal only after all the other tokens have been written.
        