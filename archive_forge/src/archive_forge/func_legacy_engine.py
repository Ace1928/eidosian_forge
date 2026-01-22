import testtools
import yaql
from yaql.language import factory
from yaql import legacy
@property
def legacy_engine(self):
    if self._legacy_engine is None:
        self._legacy_engine = self.create_legacy_engine()
    return self._legacy_engine