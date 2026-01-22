import functools
import re
from collections import defaultdict
from graphlib import TopologicalSorter
from itertools import chain
from django.conf import settings
from django.db import models
from django.db.migrations import operations
from django.db.migrations.migration import Migration
from django.db.migrations.operations.models import AlterModelOptions
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import MigrationQuestioner
from django.db.migrations.utils import (
def swappable_first_key(self, item):
    """
        Place potential swappable models first in lists of created models (only
        real way to solve #22783).
        """
    try:
        model_state = self.to_state.models[item]
        base_names = {base if isinstance(base, str) else base.__name__ for base in model_state.bases}
        string_version = '%s.%s' % (item[0], item[1])
        if model_state.options.get('swappable') or 'AbstractUser' in base_names or 'AbstractBaseUser' in base_names or (settings.AUTH_USER_MODEL.lower() == string_version.lower()):
            return ('___' + item[0], '___' + item[1])
    except LookupError:
        pass
    return item