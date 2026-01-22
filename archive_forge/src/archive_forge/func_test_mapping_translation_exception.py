import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_mapping_translation_exception(self):
    self.context.register_function(lambda *args, **kwargs: 1, name='f')
    self.assertRaises(exceptions.MappingTranslationException, self.eval, 'f(2+2 => 4)')