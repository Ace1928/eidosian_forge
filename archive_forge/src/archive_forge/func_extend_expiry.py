import contextlib
import datetime
import functools
import re
import string
import threading
import time
import fasteners
import msgpack
from oslo_serialization import msgpackutils
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from redis import exceptions as redis_exceptions
from redis import sentinel
from taskflow import exceptions as exc
from taskflow.jobs import base
from taskflow import logging
from taskflow import states
from taskflow.utils import misc
from taskflow.utils import redis_utils as ru
def extend_expiry(self, expiry):
    """Extends the owner key (aka the claim) expiry for this job.

        NOTE(harlowja): if the claim for this job did **not** previously
        have an expiry associated with it, calling this method will create
        one (and after that time elapses the claim on this job will cease
        to exist).

        Returns ``True`` if the expiry request was performed
        otherwise ``False``.
        """
    with _translate_failures():
        return ru.apply_expiry(self._client, self._owner_key, expiry, prior_version=self._redis_version)