from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
def compare_obj(test, obj, db_obj, subs=None, allow_missing=None, comparators=None):
    """Compare a VersionedObject and a dict-like database object.

    This automatically converts TZ-aware datetimes and iterates over
    the fields of the object.

    :param test: The TestCase doing the comparison
    :param obj: The VersionedObject to examine
    :param db_obj: The dict-like database object to use as reference
    :param subs: A dict of objkey=dbkey field substitutions
    :param allow_missing: A list of fields that may not be in db_obj
    :param comparators: Map of comparator functions to use for certain fields
    """
    subs = subs or {}
    allow_missing = allow_missing or []
    comparators = comparators or {}
    for key in obj.fields:
        db_key = subs.get(key, key)
        if key in allow_missing:
            if key not in obj or db_key not in db_obj:
                continue
        if not obj.obj_attr_is_set(key) and db_key not in db_obj:
            continue
        elif obj.obj_attr_is_set(key) and db_key not in db_obj:
            raise AssertionError('%s (db_key: %s) is set on the object, but not on the db_obj, so the objects are not equal' % (key, db_key))
        elif not obj.obj_attr_is_set(key) and db_key in db_obj:
            raise AssertionError('%s (db_key: %s) is set on the db_obj, but not on the object, so the objects are not equal' % (key, db_key))
        obj_val = getattr(obj, key)
        db_val = db_obj[db_key]
        if isinstance(obj_val, datetime.datetime):
            obj_val = obj_val.replace(tzinfo=None)
        if isinstance(db_val, datetime.datetime):
            db_val = obj_val.replace(tzinfo=None)
        if key in comparators:
            comparator = comparators[key]
            comparator(db_val, obj_val)
        else:
            test.assertEqual(db_val, obj_val)