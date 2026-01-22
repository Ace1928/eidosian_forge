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
def create_generic_related_manager(superclass, rel):
    """
    Factory function to create a manager that subclasses another manager
    (generally the default manager of a given model) and adds behaviors
    specific to generic relations.
    """

    class GenericRelatedObjectManager(superclass, AltersData):

        def __init__(self, instance=None):
            super().__init__()
            self.instance = instance
            self.model = rel.model
            self.get_content_type = functools.partial(ContentType.objects.db_manager(instance._state.db).get_for_model, for_concrete_model=rel.field.for_concrete_model)
            self.content_type = self.get_content_type(instance)
            self.content_type_field_name = rel.field.content_type_field_name
            self.object_id_field_name = rel.field.object_id_field_name
            self.prefetch_cache_name = rel.field.attname
            self.pk_val = instance.pk
            self.core_filters = {'%s__pk' % self.content_type_field_name: self.content_type.id, self.object_id_field_name: self.pk_val}

        def __call__(self, *, manager):
            manager = getattr(self.model, manager)
            manager_class = create_generic_related_manager(manager.__class__, rel)
            return manager_class(instance=self.instance)
        do_not_call_in_templates = True

        def __str__(self):
            return repr(self)

        def _apply_rel_filters(self, queryset):
            """
            Filter the queryset for the instance this manager is bound to.
            """
            db = self._db or router.db_for_read(self.model, instance=self.instance)
            return queryset.using(db).filter(**self.core_filters)

        def _remove_prefetched_objects(self):
            try:
                self.instance._prefetched_objects_cache.pop(self.prefetch_cache_name)
            except (AttributeError, KeyError):
                pass

        def get_queryset(self):
            try:
                return self.instance._prefetched_objects_cache[self.prefetch_cache_name]
            except (AttributeError, KeyError):
                queryset = super().get_queryset()
                return self._apply_rel_filters(queryset)

        def get_prefetch_queryset(self, instances, queryset=None):
            warnings.warn('get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() instead.', RemovedInDjango60Warning, stacklevel=2)
            if queryset is None:
                return self.get_prefetch_querysets(instances)
            return self.get_prefetch_querysets(instances, [queryset])

        def get_prefetch_querysets(self, instances, querysets=None):
            if querysets and len(querysets) != 1:
                raise ValueError('querysets argument of get_prefetch_querysets() should have a length of 1.')
            queryset = querysets[0] if querysets else super().get_queryset()
            queryset._add_hints(instance=instances[0])
            queryset = queryset.using(queryset._db or self._db)
            content_type_queries = [models.Q.create([(f'{self.content_type_field_name}__pk', content_type_id), (f'{self.object_id_field_name}__in', {obj.pk for obj in objs})]) for content_type_id, objs in itertools.groupby(sorted(instances, key=lambda obj: self.get_content_type(obj).pk), lambda obj: self.get_content_type(obj).pk)]
            query = models.Q.create(content_type_queries, connector=models.Q.OR)
            object_id_converter = instances[0]._meta.pk.to_python
            content_type_id_field_name = '%s_id' % self.content_type_field_name
            return (queryset.filter(query), lambda relobj: (object_id_converter(getattr(relobj, self.object_id_field_name)), getattr(relobj, content_type_id_field_name)), lambda obj: (obj.pk, self.get_content_type(obj).pk), False, self.prefetch_cache_name, False)

        def add(self, *objs, bulk=True):
            self._remove_prefetched_objects()
            db = router.db_for_write(self.model, instance=self.instance)

            def check_and_update_obj(obj):
                if not isinstance(obj, self.model):
                    raise TypeError("'%s' instance expected, got %r" % (self.model._meta.object_name, obj))
                setattr(obj, self.content_type_field_name, self.content_type)
                setattr(obj, self.object_id_field_name, self.pk_val)
            if bulk:
                pks = []
                for obj in objs:
                    if obj._state.adding or obj._state.db != db:
                        raise ValueError("%r instance isn't saved. Use bulk=False or save the object first." % obj)
                    check_and_update_obj(obj)
                    pks.append(obj.pk)
                self.model._base_manager.using(db).filter(pk__in=pks).update(**{self.content_type_field_name: self.content_type, self.object_id_field_name: self.pk_val})
            else:
                with transaction.atomic(using=db, savepoint=False):
                    for obj in objs:
                        check_and_update_obj(obj)
                        obj.save()
        add.alters_data = True

        async def aadd(self, *objs, bulk=True):
            return await sync_to_async(self.add)(*objs, bulk=bulk)
        aadd.alters_data = True

        def remove(self, *objs, bulk=True):
            if not objs:
                return
            self._clear(self.filter(pk__in=[o.pk for o in objs]), bulk)
        remove.alters_data = True

        async def aremove(self, *objs, bulk=True):
            return await sync_to_async(self.remove)(*objs, bulk=bulk)
        aremove.alters_data = True

        def clear(self, *, bulk=True):
            self._clear(self, bulk)
        clear.alters_data = True

        async def aclear(self, *, bulk=True):
            return await sync_to_async(self.clear)(bulk=bulk)
        aclear.alters_data = True

        def _clear(self, queryset, bulk):
            self._remove_prefetched_objects()
            db = router.db_for_write(self.model, instance=self.instance)
            queryset = queryset.using(db)
            if bulk:
                queryset.delete()
            else:
                with transaction.atomic(using=db, savepoint=False):
                    for obj in queryset:
                        obj.delete()
        _clear.alters_data = True

        def set(self, objs, *, bulk=True, clear=False):
            objs = tuple(objs)
            db = router.db_for_write(self.model, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                if clear:
                    self.clear()
                    self.add(*objs, bulk=bulk)
                else:
                    old_objs = set(self.using(db).all())
                    new_objs = []
                    for obj in objs:
                        if obj in old_objs:
                            old_objs.remove(obj)
                        else:
                            new_objs.append(obj)
                    self.remove(*old_objs)
                    self.add(*new_objs, bulk=bulk)
        set.alters_data = True

        async def aset(self, objs, *, bulk=True, clear=False):
            return await sync_to_async(self.set)(objs, bulk=bulk, clear=clear)
        aset.alters_data = True

        def create(self, **kwargs):
            self._remove_prefetched_objects()
            kwargs[self.content_type_field_name] = self.content_type
            kwargs[self.object_id_field_name] = self.pk_val
            db = router.db_for_write(self.model, instance=self.instance)
            return super().using(db).create(**kwargs)
        create.alters_data = True

        async def acreate(self, **kwargs):
            return await sync_to_async(self.create)(**kwargs)
        acreate.alters_data = True

        def get_or_create(self, **kwargs):
            kwargs[self.content_type_field_name] = self.content_type
            kwargs[self.object_id_field_name] = self.pk_val
            db = router.db_for_write(self.model, instance=self.instance)
            return super().using(db).get_or_create(**kwargs)
        get_or_create.alters_data = True

        async def aget_or_create(self, **kwargs):
            return await sync_to_async(self.get_or_create)(**kwargs)
        aget_or_create.alters_data = True

        def update_or_create(self, **kwargs):
            kwargs[self.content_type_field_name] = self.content_type
            kwargs[self.object_id_field_name] = self.pk_val
            db = router.db_for_write(self.model, instance=self.instance)
            return super().using(db).update_or_create(**kwargs)
        update_or_create.alters_data = True

        async def aupdate_or_create(self, **kwargs):
            return await sync_to_async(self.update_or_create)(**kwargs)
        aupdate_or_create.alters_data = True
    return GenericRelatedObjectManager