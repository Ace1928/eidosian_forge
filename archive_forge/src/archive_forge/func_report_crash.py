import os
import getpass
from socket import gethostname
import sys
import uuid
from time import strftime
from traceback import format_exception
from ... import logging
from ...utils.filemanip import savepkl, crash2txt
import sys
import os
from nipype import config, logging
from nipype.utils.filemanip import loadpkl, savepkl
from socket import gethostname
from traceback import format_exception
from nipype.utils.filemanip import loadpkl, savepkl
def report_crash(node, traceback=None, hostname=None):
    """Writes crash related information to a file"""
    name = node._id
    host = None
    traceback = traceback or format_exception(*sys.exc_info())
    try:
        result = node.result
    except FileNotFoundError:
        traceback += '\n\nWhen creating this crashfile, the results file corresponding\nto the node could not be found.'.splitlines(keepends=True)
    except Exception as exc:
        traceback += '\n\nDuring the creation of this crashfile triggered by the above exception,\nanother exception occurred:\n\n{}.'.format(exc).splitlines(keepends=True)
    else:
        if getattr(result, 'runtime', None):
            if isinstance(result.runtime, list):
                host = result.runtime[0].hostname
            else:
                host = result.runtime.hostname
    host = host or hostname or gethostname()
    logger.error('Node %s failed to run on host %s.', name, host)
    timeofcrash = strftime('%Y%m%d-%H%M%S')
    try:
        login_name = getpass.getuser()
    except KeyError:
        login_name = 'UID{:d}'.format(os.getuid())
    crashfile = 'crash-%s-%s-%s-%s' % (timeofcrash, login_name, name, str(uuid.uuid4()))
    crashdir = node.config['execution'].get('crashdump_dir', os.getcwd())
    os.makedirs(crashdir, exist_ok=True)
    crashfile = os.path.join(crashdir, crashfile)
    if node.config['execution']['crashfile_format'].lower() in ('text', 'txt', '.txt'):
        crashfile += '.txt'
    else:
        crashfile += '.pklz'
    logger.error('Saving crash info to %s\n%s', crashfile, ''.join(traceback))
    if crashfile.endswith('.txt'):
        crash2txt(crashfile, dict(node=node, traceback=traceback))
    else:
        savepkl(crashfile, dict(node=node, traceback=traceback), versioning=True)
    return crashfile