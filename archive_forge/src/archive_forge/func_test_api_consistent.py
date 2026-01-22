import pytest
from modin.core.storage_formats import BaseQueryCompiler, PandasQueryCompiler
@pytest.mark.parametrize('execution', EXECUTIONS)
def test_api_consistent(execution):
    base_methods = set(BASE_EXECUTION.__dict__)
    custom_methods = set([key for key in execution.__dict__.keys() if not key.startswith('_')])
    extra_methods = custom_methods.difference(base_methods)
    assert len(extra_methods) == 0, f'{execution} implement these extra methods: {extra_methods}'