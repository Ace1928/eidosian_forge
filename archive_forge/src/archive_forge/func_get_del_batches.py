from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
def get_del_batches(self, objs, fields):
    """
        Return the objs in suitably sized batches for the used connection.
        """
    field_names = [field.name for field in fields]
    conn_batch_size = max(connections[self.using].ops.bulk_batch_size(field_names, objs), 1)
    if len(objs) > conn_batch_size:
        return [objs[i:i + conn_batch_size] for i in range(0, len(objs), conn_batch_size)]
    else:
        return [objs]