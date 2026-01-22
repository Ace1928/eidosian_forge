from collections import defaultdict
from logging import getLogger
from typing import Any, DefaultDict
from pip._vendor.resolvelib.reporters import BaseReporter
from .base import Candidate, Requirement
def starting_round(self, index: int) -> None:
    logger.info('Reporter.starting_round(%r)', index)