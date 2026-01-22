import re
from django.contrib.gis.db import models
@property
def supports_collect_aggr(self):
    return models.Collect not in self.connection.ops.disallowed_aggregates