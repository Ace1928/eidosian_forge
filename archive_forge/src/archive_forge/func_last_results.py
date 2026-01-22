import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
@property
def last_results(self):
    """The last result that was produced."""
    try:
        return self.results[-1][0]
    except IndexError:
        exc.raise_with_cause(exc.NotFound, 'Last results not found')