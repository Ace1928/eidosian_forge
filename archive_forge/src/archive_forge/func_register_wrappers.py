from types import NoneType
from django.db.backends.utils import names_digest, split_identifier
from django.db.models.expressions import Col, ExpressionList, F, Func, OrderBy
from django.db.models.functions import Collate
from django.db.models.query_utils import Q
from django.db.models.sql import Query
from django.utils.functional import partition
@classmethod
def register_wrappers(cls, *wrapper_classes):
    cls.wrapper_classes = wrapper_classes