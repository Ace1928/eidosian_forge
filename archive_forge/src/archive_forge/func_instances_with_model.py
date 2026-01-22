from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
def instances_with_model(self):
    for model, instances in self.data.items():
        for obj in instances:
            yield (model, obj)