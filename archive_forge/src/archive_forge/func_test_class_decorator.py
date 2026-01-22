import collections
import logging
import pytest
import modin.logging
from modin.config import LogMode
def test_class_decorator(monkeypatch, get_log_messages):

    @modin.logging.enable_logging('CUSTOM')
    class Foo:

        def method1(self):
            pass

        @classmethod
        def method2(cls):
            pass

        @staticmethod
        def method3():
            pass

    class Bar(Foo):

        def method4(self):
            pass
    with monkeypatch.context() as ctx:
        mock_get_logger(ctx)
        Foo().method1()
        Foo.method2()
        Foo.method3()
        Bar().method1()
        Bar().method4()
    assert get_log_messages()[logging.INFO] == ['START::CUSTOM::Foo.method1', 'STOP::CUSTOM::Foo.method1', 'START::CUSTOM::Foo.method2', 'STOP::CUSTOM::Foo.method2', 'START::CUSTOM::Foo.method3', 'STOP::CUSTOM::Foo.method3', 'START::CUSTOM::Foo.method1', 'STOP::CUSTOM::Foo.method1']