import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'nvcc --version', 'release .+ V(.*)')