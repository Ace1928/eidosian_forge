import re
from django.contrib.gis.db import models
@property
def supports_union_aggr(self):
    return models.Union not in self.connection.ops.disallowed_aggregates