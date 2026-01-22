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
def check_dependency(self, operation, dependency):
    """
        Return True if the given operation depends on the given dependency,
        False otherwise.
        """
    if dependency[2] is None and dependency[3] is True:
        return isinstance(operation, operations.CreateModel) and operation.name_lower == dependency[1].lower()
    elif dependency[2] is not None and dependency[3] is True:
        return isinstance(operation, operations.CreateModel) and operation.name_lower == dependency[1].lower() and any((dependency[2] == x for x, y in operation.fields)) or (isinstance(operation, operations.AddField) and operation.model_name_lower == dependency[1].lower() and (operation.name_lower == dependency[2].lower()))
    elif dependency[2] is not None and dependency[3] is False:
        return isinstance(operation, operations.RemoveField) and operation.model_name_lower == dependency[1].lower() and (operation.name_lower == dependency[2].lower())
    elif dependency[2] is None and dependency[3] is False:
        return isinstance(operation, operations.DeleteModel) and operation.name_lower == dependency[1].lower()
    elif dependency[2] is not None and dependency[3] == 'alter':
        return isinstance(operation, operations.AlterField) and operation.model_name_lower == dependency[1].lower() and (operation.name_lower == dependency[2].lower())
    elif dependency[2] is not None and dependency[3] == 'order_wrt_unset':
        return isinstance(operation, operations.AlterOrderWithRespectTo) and operation.name_lower == dependency[1].lower() and ((operation.order_with_respect_to or '').lower() != dependency[2].lower())
    elif dependency[2] is not None and dependency[3] == 'foo_together_change':
        return isinstance(operation, (operations.AlterUniqueTogether, operations.AlterIndexTogether)) and operation.name_lower == dependency[1].lower()
    else:
        raise ValueError("Can't handle dependency %r" % (dependency,))