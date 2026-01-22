from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
def get_for_id(self, id):
    """
        Lookup a ContentType by ID. Use the same shared cache as get_for_model
        (though ContentTypes are not created on-the-fly by get_by_id).
        """
    try:
        ct = self._cache[self.db][id]
    except KeyError:
        ct = self.get(pk=id)
        self._add_to_cache(self.db, ct)
    return ct