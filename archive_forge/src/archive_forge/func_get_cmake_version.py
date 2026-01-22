import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cmake --version', 'cmake (.*)')