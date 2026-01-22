from django.db import models
from django.utils.translation import gettext_lazy as _
@classmethod
def get_session_store_class(cls):
    raise NotImplementedError