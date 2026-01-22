from django.db.models import Prefetch
from django.db.models.query import ModelIterable, RawQuerySet
def get_current_querysets(self, level):
    if self.get_current_prefetch_to(level) == self.prefetch_to:
        return self.querysets
    return None