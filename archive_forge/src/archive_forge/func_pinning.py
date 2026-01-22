from collections import defaultdict
from logging import getLogger
from typing import Any, DefaultDict
from pip._vendor.resolvelib.reporters import BaseReporter
from .base import Candidate, Requirement
def pinning(self, candidate: Candidate) -> None:
    logger.info('Reporter.pinning(%r)', candidate)