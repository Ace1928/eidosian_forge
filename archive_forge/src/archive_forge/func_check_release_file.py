import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def check_release_file(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cat /etc/*-release', 'PRETTY_NAME="(.*)"')