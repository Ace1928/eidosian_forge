import logging
from django.contrib.sessions.backends.base import CreateError, SessionBase, UpdateError
from django.core.exceptions import SuspiciousOperation
from django.db import DatabaseError, IntegrityError, router, transaction
from django.utils import timezone
from django.utils.functional import cached_property
@classmethod
def get_model_class(cls):
    from django.contrib.sessions.models import Session
    return Session