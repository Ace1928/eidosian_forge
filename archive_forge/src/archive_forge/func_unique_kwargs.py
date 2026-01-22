import sys
from decimal import Decimal
from decimal import InvalidOperation as DecimalInvalidOperation
from pathlib import Path
from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.gdal import (
from django.contrib.gis.gdal.field import (
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import connections, models, router, transaction
from django.utils.encoding import force_str
def unique_kwargs(self, kwargs):
    """
        Given the feature keyword arguments (from `feature_kwargs`), construct
        and return the uniqueness keyword arguments -- a subset of the feature
        kwargs.
        """
    if isinstance(self.unique, str):
        return {self.unique: kwargs[self.unique]}
    else:
        return {fld: kwargs[fld] for fld in self.unique}