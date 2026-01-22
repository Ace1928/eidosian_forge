import os
import re
from time import sleep
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
LSF lists a status of 'PEND' when a job has been submitted but is
        waiting to be picked up, and 'RUN' when it is actively being processed.
        But _is_pending should return True until a job has finished and is
        ready to be checked for completeness. So return True if status is
        either 'PEND' or 'RUN'