from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
def contribute_to_class(self, cls, name, **kwargs):
    super().contribute_to_class(cls, name, **kwargs)
    setattr(cls, self.attname, SpatialProxy(gdal.GDALRaster, self))