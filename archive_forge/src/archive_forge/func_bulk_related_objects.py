import functools
import itertools
import warnings
from collections import defaultdict
from asgiref.sync import sync_to_async
from django.contrib.contenttypes.models import ContentType
from django.core import checks
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import DEFAULT_DB_ALIAS, models, router, transaction
from django.db.models import DO_NOTHING, ForeignObject, ForeignObjectRel
from django.db.models.base import ModelBase, make_foreign_order_accessors
from django.db.models.fields.mixins import FieldCacheMixin
from django.db.models.fields.related import (
from django.db.models.query_utils import PathInfo
from django.db.models.sql import AND
from django.db.models.sql.where import WhereNode
from django.db.models.utils import AltersData
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
def bulk_related_objects(self, objs, using=DEFAULT_DB_ALIAS):
    """
        Return all objects related to ``objs`` via this ``GenericRelation``.
        """
    return self.remote_field.model._base_manager.db_manager(using).filter(**{'%s__pk' % self.content_type_field_name: ContentType.objects.db_manager(using).get_for_model(self.model, for_concrete_model=self.for_concrete_model).pk, '%s__in' % self.object_id_field_name: [obj.pk for obj in objs]})