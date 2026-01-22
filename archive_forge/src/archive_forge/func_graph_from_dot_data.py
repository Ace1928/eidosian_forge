import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def graph_from_dot_data(s):
    """Load graphs from DOT description in string `s`.

    @param s: string in [DOT language](
        https://en.wikipedia.org/wiki/DOT_(graph_description_language))

    @return: Graphs that result from parsing.
    @rtype: `list` of `pydot.Dot`
    """
    return dot_parser.parse_dot_data(s)