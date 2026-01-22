import contextlib
import re
import sys
def assert_raises_message_with_given_cause(except_cls, msg, cause_cls, callable_, *args, **kwargs):
    return _assert_raises(except_cls, callable_, args, kwargs, msg=msg, cause_cls=cause_cls)