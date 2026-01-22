from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.db.models import Exists, OuterRef, Q
def configure_user(self, request, user, created=True):
    """
        Configure a user and return the updated user.

        By default, return the user unmodified.
        """
    return user