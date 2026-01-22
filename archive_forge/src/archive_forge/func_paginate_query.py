import collections
from collections import abc
import itertools
import logging
import re
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import Boolean
from sqlalchemy.engine import Connectable
from sqlalchemy.engine import url as sa_url
from sqlalchemy import exc
from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql.expression import literal_column
from sqlalchemy.sql import text
from sqlalchemy import Table
from oslo_db._i18n import _
from oslo_db import exception
from oslo_db.sqlalchemy import models
def paginate_query(query, model, limit, sort_keys, marker=None, sort_dir=None, sort_dirs=None):
    """Returns a query with sorting / pagination criteria added.

    Pagination works by requiring a unique sort_key, specified by sort_keys.
    (If sort_keys is not unique, then we risk looping through values.)
    We use the last row in the previous page as the 'marker' for pagination.
    So we must return values that follow the passed marker in the order.
    With a single-valued sort_key, this would be easy: sort_key > X.
    With a compound-values sort_key, (k1, k2, k3) we must do this to repeat
    the lexicographical ordering:
    (k1 > X1) or (k1 == X1 && k2 > X2) or (k1 == X1 && k2 == X2 && k3 > X3)

    We also have to cope with different sort_directions and cases where k2,
    k3, ... are nullable.

    Typically, the id of the last row is used as the client-facing pagination
    marker, then the actual marker object must be fetched from the db and
    passed in to us as marker.

    The "offset" parameter is intentionally avoided. As offset requires a
    full scan through the preceding results each time, criteria-based
    pagination is preferred. See http://use-the-index-luke.com/no-offset
    for further background.

    :param query: the query object to which we should add paging/sorting
    :param model: the ORM model class
    :param limit: maximum number of items to return
    :param sort_keys: array of attributes by which results should be sorted
    :param marker: the last item of the previous page; we returns the next
                    results after this value.
    :param sort_dir: direction in which results should be sorted (asc, desc)
                     suffix -nullsfirst, -nullslast can be added to defined
                     the ordering of null values
    :param sort_dirs: per-column array of sort_dirs, corresponding to sort_keys

    :rtype: sqlalchemy.orm.query.Query
    :return: The query with sorting/pagination added.
    """
    if _stable_sorting_order(model, sort_keys) is False:
        LOG.warning('Unique keys not in sort_keys. The sorting order may be unstable.')
    if sort_dir and sort_dirs:
        raise AssertionError('Disallow set sort_dir and sort_dirs at the same time.')
    if sort_dirs is None and sort_dir is None:
        sort_dir = 'asc'
    if sort_dirs is None:
        sort_dirs = [sort_dir for _sort_key in sort_keys]
    if len(sort_dirs) != len(sort_keys):
        raise AssertionError('sort_dirs and sort_keys must have same length.')
    for current_sort_key, current_sort_dir in zip(sort_keys, sort_dirs):
        try:
            inspect(model).all_orm_descriptors[current_sort_key]
        except KeyError:
            raise exception.InvalidSortKey(current_sort_key)
        else:
            sort_key_attr = getattr(model, current_sort_key)
        try:
            main_sort_dir, __, null_sort_dir = current_sort_dir.partition('-')
            sort_dir_func = {'asc': sqlalchemy.asc, 'desc': sqlalchemy.desc}[main_sort_dir]
            null_order_by_stmt = {'': None, 'nullsfirst': sort_key_attr.is_(None), 'nullslast': sort_key_attr.is_not(None)}[null_sort_dir]
        except KeyError:
            raise ValueError(_('Unknown sort direction, must be one of: %s') % ', '.join(_VALID_SORT_DIR))
        if null_order_by_stmt is not None:
            query = query.order_by(sqlalchemy.desc(null_order_by_stmt))
        query = query.order_by(sort_dir_func(sort_key_attr))
    if marker is not None:
        marker_values = []
        for sort_key in sort_keys:
            v = getattr(marker, sort_key)
            marker_values.append(v)
        criteria_list = []
        for i in range(len(sort_keys)):
            crit_attrs = []
            if marker_values[i] is not None:
                for j in range(i):
                    model_attr = getattr(model, sort_keys[j])
                    if marker_values[j] is not None:
                        crit_attrs.append(model_attr == marker_values[j])
                model_attr = getattr(model, sort_keys[i])
                val = marker_values[i]
                if isinstance(model_attr.type, Boolean):
                    val = int(val)
                    model_attr = cast(model_attr, Integer)
                if sort_dirs[i].startswith('desc'):
                    crit_attr = model_attr < val
                    if sort_dirs[i].endswith('nullsfirst'):
                        crit_attr = sqlalchemy.sql.or_(crit_attr, model_attr.is_(None))
                else:
                    crit_attr = model_attr > val
                    if sort_dirs[i].endswith('nullslast'):
                        crit_attr = sqlalchemy.sql.or_(crit_attr, model_attr.is_(None))
                crit_attrs.append(crit_attr)
                criteria = sqlalchemy.sql.and_(*crit_attrs)
                criteria_list.append(criteria)
        f = sqlalchemy.sql.or_(*criteria_list)
        query = query.filter(f)
    if limit is not None:
        query = query.limit(limit)
    return query