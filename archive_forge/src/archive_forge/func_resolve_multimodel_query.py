import threading
from peewee import *
from peewee import Alias
from peewee import CompoundSelectQuery
from peewee import Metadata
from peewee import callable_
from peewee import __deprecated__
def resolve_multimodel_query(query, key='_model_identifier'):
    mapping = {}
    accum = [query]
    while accum:
        curr = accum.pop()
        if isinstance(curr, CompoundSelectQuery):
            accum.extend((curr.lhs, curr.rhs))
            continue
        model_class = curr.model
        name = model_class._meta.table_name
        mapping[name] = model_class
        curr._returning.append(Value(name).alias(key))

    def wrapped_iterator():
        for row in query.dicts().iterator():
            identifier = row.pop(key)
            model = mapping[identifier]
            yield model(**row)
    return wrapped_iterator()