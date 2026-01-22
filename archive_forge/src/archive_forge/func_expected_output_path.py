import keyword
import os
import re
import subprocess
import sys
from taskflow import test
def expected_output_path(name):
    return root_path('taskflow', 'examples', '%s.out.txt' % name)