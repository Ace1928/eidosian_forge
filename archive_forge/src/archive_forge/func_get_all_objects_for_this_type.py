from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
def get_all_objects_for_this_type(self, **kwargs):
    """
        Return all objects of this type for the keyword arguments given.
        """
    return self.model_class()._base_manager.using(self._state.db).filter(**kwargs)