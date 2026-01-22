from __future__ import absolute_import, division, print_function
import errno
import json
import shlex
import shutil
import os
import subprocess
import sys
import traceback
import signal
import time
import syslog
import multiprocessing
from ansible.module_utils.common.text.converters import to_text, to_bytes
def jwrite(info):
    jobfile = job_path + '.tmp'
    tjob = open(jobfile, 'w')
    try:
        tjob.write(json.dumps(info))
    except (IOError, OSError) as e:
        notice('failed to write to %s: %s' % (jobfile, str(e)))
        raise e
    finally:
        tjob.close()
        os.rename(jobfile, job_path)