import operator
from django.db import DataError, InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.utils.functional import cached_property
@cached_property
def django_test_expected_failures(self):
    expected_failures = set()
    if self.uses_server_side_binding:
        expected_failures.update({'aggregation.tests.AggregateTestCase.test_group_by_nested_expression_with_params'})
    return expected_failures