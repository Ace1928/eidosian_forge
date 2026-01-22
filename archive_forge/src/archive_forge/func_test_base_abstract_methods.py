import pytest
from modin.core.storage_formats import BaseQueryCompiler, PandasQueryCompiler
def test_base_abstract_methods():
    allowed_abstract_methods = ['__init__', 'free', 'finalize', 'execute', 'to_pandas', 'from_pandas', 'from_arrow', 'default_to_pandas', 'from_dataframe', 'to_dataframe']
    not_implemented_methods = BASE_EXECUTION.__abstractmethods__.difference(allowed_abstract_methods)
    not_implemented_methods = list(not_implemented_methods)
    not_implemented_methods.sort()
    assert len(not_implemented_methods) == 0, f'{BASE_EXECUTION} has not implemented abstract methods: {not_implemented_methods}'