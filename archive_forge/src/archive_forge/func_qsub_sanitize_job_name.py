import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
def qsub_sanitize_job_name(testjobname):
    """Ensure that qsub job names must begin with a letter.

    Numbers and punctuation are  not allowed.

    >>> qsub_sanitize_job_name('01')
    'J01'
    >>> qsub_sanitize_job_name('a01')
    'a01'
    """
    if testjobname[0].isalpha():
        return testjobname
    else:
        return 'J' + testjobname