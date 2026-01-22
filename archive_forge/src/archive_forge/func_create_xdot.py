import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def create_xdot(dotdata, prog='dot', options=''):
    """Run a graph through Graphviz and return an xdot-version of the graph"""
    progs = dotparsing.find_graphviz()
    if progs is None:
        log.error('Could not locate Graphviz binaries')
        return None
    if not prog in progs:
        log.error('Invalid prog=%s', prog)
        raise NameError('The %s program is not recognized. Valid values are %s' % (prog, list(progs)))
    tmp_fd, tmp_name = tempfile.mkstemp()
    os.close(tmp_fd)
    if os.sys.version_info[0] >= 3:
        with open(tmp_name, 'w', encoding='utf8') as f:
            f.write(dotdata)
    else:
        with open(tmp_name, 'w') as f:
            f.write(dotdata)
    output_format = 'xdot'
    progpath = '"%s"' % progs[prog].strip()
    cmd = progpath + ' -T' + output_format + ' ' + options + ' ' + tmp_name
    log.debug('Creating xdot data with: %s', cmd)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, close_fds=sys.platform != 'win32')
    stdout, stderr = (p.stdout, p.stderr)
    try:
        data = stdout.read()
    finally:
        stdout.close()
    try:
        error_data = stderr.read()
        if error_data:
            if b'Error:' in error_data:
                log.error('Graphviz returned with the following message: %s', error_data)
            else:
                log.debug('Graphviz STDERR %s', error_data)
    finally:
        stderr.close()
    p.kill()
    p.wait()
    os.unlink(tmp_name)
    return data