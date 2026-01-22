from django.contrib.gis.db.backends.base.models import SpatialRefSysMixin
from django.db import models
@property
def wkt(self):
    return self.srtext