from collections import defaultdict
from logging import getLogger
from typing import Any, DefaultDict
from pip._vendor.resolvelib.reporters import BaseReporter
from .base import Candidate, Requirement
def rejecting_candidate(self, criterion: Any, candidate: Candidate) -> None:
    logger.info('Reporter.rejecting_candidate(%r, %r)', criterion, candidate)