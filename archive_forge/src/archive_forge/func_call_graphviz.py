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
def call_graphviz(program, arguments, working_dir, **kwargs):
    if program in DEFAULT_PROGRAMS:
        extension = get_executable_extension()
        program += extension
    if arguments is None:
        arguments = []
    env = {'PATH': os.environ.get('PATH', ''), 'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''), 'SYSTEMROOT': os.environ.get('SYSTEMROOT', '')}
    program_with_args = [program] + arguments
    process = subprocess.Popen(program_with_args, env=env, cwd=working_dir, shell=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE, **kwargs)
    stdout_data, stderr_data = process.communicate()
    return (stdout_data, stderr_data, process)