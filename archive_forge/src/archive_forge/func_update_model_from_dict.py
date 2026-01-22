import threading
from peewee import *
from peewee import Alias
from peewee import CompoundSelectQuery
from peewee import Metadata
from peewee import callable_
from peewee import __deprecated__
def update_model_from_dict(instance, data, ignore_unknown=False):
    meta = instance._meta
    backrefs = dict([(fk.backref, fk) for fk in meta.backrefs])
    for key, value in data.items():
        if key in meta.combined:
            field = meta.combined[key]
            is_backref = False
        elif key in backrefs:
            field = backrefs[key]
            is_backref = True
        elif ignore_unknown:
            setattr(instance, key, value)
            continue
        else:
            raise AttributeError('Unrecognized attribute "%s" for model class %s.' % (key, type(instance)))
        is_foreign_key = isinstance(field, ForeignKeyField)
        if not is_backref and is_foreign_key and isinstance(value, dict):
            try:
                rel_instance = instance.__rel__[field.name]
            except KeyError:
                rel_instance = field.rel_model()
            setattr(instance, field.name, update_model_from_dict(rel_instance, value, ignore_unknown))
        elif is_backref and isinstance(value, (list, tuple)):
            instances = [dict_to_model(field.model, row_data, ignore_unknown) for row_data in value]
            for rel_instance in instances:
                setattr(rel_instance, field.name, instance)
            setattr(instance, field.backref, instances)
        else:
            setattr(instance, field.name, value)
    return instance