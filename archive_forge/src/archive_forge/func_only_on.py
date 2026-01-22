import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def only_on(dbs, reason=None):
    return only_if(OrPredicate([Predicate.as_predicate(db, reason) for db in util.to_list(dbs)]))