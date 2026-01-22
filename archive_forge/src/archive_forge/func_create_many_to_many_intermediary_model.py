import functools
import inspect
import warnings
from functools import partial
from django import forms
from django.apps import apps
from django.conf import SettingsReference, settings
from django.core import checks, exceptions
from django.db import connection, router
from django.db.backends import utils
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import CASCADE, SET_DEFAULT, SET_NULL
from django.db.models.query_utils import PathInfo
from django.db.models.utils import make_model_tuple
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import FieldCacheMixin
from .related_descriptors import (
from .related_lookups import (
from .reverse_related import ForeignObjectRel, ManyToManyRel, ManyToOneRel, OneToOneRel
def create_many_to_many_intermediary_model(field, klass):
    from django.db import models

    def set_managed(model, related, through):
        through._meta.managed = model._meta.managed or related._meta.managed
    to_model = resolve_relation(klass, field.remote_field.model)
    name = '%s_%s' % (klass._meta.object_name, field.name)
    lazy_related_operation(set_managed, klass, to_model, name)
    to = make_model_tuple(to_model)[1]
    from_ = klass._meta.model_name
    if to == from_:
        to = 'to_%s' % to
        from_ = 'from_%s' % from_
    meta = type('Meta', (), {'db_table': field._get_m2m_db_table(klass._meta), 'auto_created': klass, 'app_label': klass._meta.app_label, 'db_tablespace': klass._meta.db_tablespace, 'unique_together': (from_, to), 'verbose_name': _('%(from)s-%(to)s relationship') % {'from': from_, 'to': to}, 'verbose_name_plural': _('%(from)s-%(to)s relationships') % {'from': from_, 'to': to}, 'apps': field.model._meta.apps})
    return type(name, (models.Model,), {'Meta': meta, '__module__': klass.__module__, from_: models.ForeignKey(klass, related_name='%s+' % name, db_tablespace=field.db_tablespace, db_constraint=field.remote_field.db_constraint, on_delete=CASCADE), to: models.ForeignKey(to_model, related_name='%s+' % name, db_tablespace=field.db_tablespace, db_constraint=field.remote_field.db_constraint, on_delete=CASCADE)})