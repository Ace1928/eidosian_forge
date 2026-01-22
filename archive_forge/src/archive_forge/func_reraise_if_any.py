import collections
import copy
import io
import os
import sys
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from taskflow import exceptions as exc
from taskflow.utils import iter_utils
from taskflow.utils import schema_utils as su
@staticmethod
def reraise_if_any(failures):
    """Re-raise exceptions if argument is not empty.

        If argument is empty list/tuple/iterator, this method returns
        None. If argument is converted into a list with a
        single ``Failure`` object in it, that failure is reraised. Else, a
        :class:`~taskflow.exceptions.WrappedFailure` exception
        is raised with the failure list as causes.
        """
    if not isinstance(failures, (list, tuple)):
        failures = list(failures)
    if len(failures) == 1:
        failures[0].reraise()
    elif len(failures) > 1:
        raise exc.WrappedFailure(failures)