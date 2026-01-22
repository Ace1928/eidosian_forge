import doctest
import os
import pickle
import sys
from tempfile import mkstemp
import unittest
from genshi.core import Markup
from genshi.template.base import Context
from genshi.template.eval import Expression, Suite, Undefined, UndefinedError, \
from genshi.compat import BytesIO, IS_PYTHON2, wrapped_bytes
def test_error_getitem_undefined_string(self):

    class Something(object):

        def __repr__(self):
            return '<Something>'
    expr = Expression('something["nil"]', filename='index.html', lineno=50, lookup='strict')
    try:
        expr.evaluate({'something': Something()})
        self.fail('Expected UndefinedError')
    except UndefinedError as e:
        self.assertEqual('<Something> has no member named "nil"', str(e))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        search_string = '<Expression \'something["nil"]\'>'
        frame = exc_traceback.tb_next
        while frame.tb_next:
            frame = frame.tb_next
            code = frame.tb_frame.f_code
            if code.co_name == search_string:
                break
        else:
            self.fail('never found the frame I was looking for')
        self.assertEqual('index.html', code.co_filename)
        self.assertEqual(50, frame.tb_lineno)