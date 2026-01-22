import unittest
from traits.api import (
from traits.observation.api import (
def test_warning_on_handler_with_bad_signature(self):
    message_regex = 'should be callable with a single positional argument'
    with self.assertWarnsRegex(UserWarning, message_regex):

        class A(HasTraits):
            foo = Int()

            @observe('foo')
            def _do_something_when_foo_changes(self):
                pass
    with self.assertWarnsRegex(UserWarning, message_regex):

        class B(HasTraits):
            foo = Int()

            @observe('foo')
            def _do_something_when_foo_changes(self, **kwargs):
                pass