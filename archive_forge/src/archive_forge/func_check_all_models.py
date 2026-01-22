import inspect
import types
from collections import defaultdict
from itertools import chain
from django.apps import apps
from django.conf import settings
from django.core.checks import Error, Tags, Warning, register
@register(Tags.models)
def check_all_models(app_configs=None, **kwargs):
    db_table_models = defaultdict(list)
    indexes = defaultdict(list)
    constraints = defaultdict(list)
    errors = []
    if app_configs is None:
        models = apps.get_models()
    else:
        models = chain.from_iterable((app_config.get_models() for app_config in app_configs))
    for model in models:
        if model._meta.managed and (not model._meta.proxy):
            db_table_models[model._meta.db_table].append(model._meta.label)
        if not inspect.ismethod(model.check):
            errors.append(Error("The '%s.check()' class method is currently overridden by %r." % (model.__name__, model.check), obj=model, id='models.E020'))
        else:
            errors.extend(model.check(**kwargs))
        for model_index in model._meta.indexes:
            indexes[model_index.name].append(model._meta.label)
        for model_constraint in model._meta.constraints:
            constraints[model_constraint.name].append(model._meta.label)
    if settings.DATABASE_ROUTERS:
        error_class, error_id = (Warning, 'models.W035')
        error_hint = 'You have configured settings.DATABASE_ROUTERS. Verify that %s are correctly routed to separate databases.'
    else:
        error_class, error_id = (Error, 'models.E028')
        error_hint = None
    for db_table, model_labels in db_table_models.items():
        if len(model_labels) != 1:
            model_labels_str = ', '.join(model_labels)
            errors.append(error_class("db_table '%s' is used by multiple models: %s." % (db_table, model_labels_str), obj=db_table, hint=error_hint % model_labels_str if error_hint else None, id=error_id))
    for index_name, model_labels in indexes.items():
        if len(model_labels) > 1:
            model_labels = set(model_labels)
            errors.append(Error("index name '%s' is not unique %s %s." % (index_name, 'for model' if len(model_labels) == 1 else 'among models:', ', '.join(sorted(model_labels))), id='models.E029' if len(model_labels) == 1 else 'models.E030'))
    for constraint_name, model_labels in constraints.items():
        if len(model_labels) > 1:
            model_labels = set(model_labels)
            errors.append(Error("constraint name '%s' is not unique %s %s." % (constraint_name, 'for model' if len(model_labels) == 1 else 'among models:', ', '.join(sorted(model_labels))), id='models.E031' if len(model_labels) == 1 else 'models.E032'))
    return errors