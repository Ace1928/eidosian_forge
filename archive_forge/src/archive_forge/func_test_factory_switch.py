from contextlib import contextmanager
import pytest
import modin.pandas as pd
from modin import set_execution
from modin.config import Engine, Parameter, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.core.execution.dispatching.factories.dispatcher import (
from modin.core.execution.python.implementations.pandas_on_python.io import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
def test_factory_switch():
    with _switch_execution('Python', 'Pandas'):
        with _switch_value(Engine, 'Test'):
            assert FactoryDispatcher.get_factory() == PandasOnTestFactory
            assert FactoryDispatcher.get_factory().io_cls == 'Foo'
        with _switch_value(StorageFormat, 'Test'):
            assert FactoryDispatcher.get_factory() == TestOnPythonFactory
            assert FactoryDispatcher.get_factory().io_cls == 'Bar'