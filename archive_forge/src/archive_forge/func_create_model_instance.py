import logging
from django.contrib.sessions.backends.base import CreateError, SessionBase, UpdateError
from django.core.exceptions import SuspiciousOperation
from django.db import DatabaseError, IntegrityError, router, transaction
from django.utils import timezone
from django.utils.functional import cached_property
def create_model_instance(self, data):
    """
        Return a new instance of the session model object, which represents the
        current session state. Intended to be used for saving the session data
        to the database.
        """
    return self.model(session_key=self._get_or_create_session_key(), session_data=self.encode(data), expire_date=self.get_expiry_date())