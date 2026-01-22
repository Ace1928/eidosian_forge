import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_print_():
    save = sys.stdout
    out = sys.stdout = six.moves.StringIO()
    try:
        six.print_('Hello,', 'person!')
    finally:
        sys.stdout = save
    assert out.getvalue() == 'Hello, person!\n'
    out = six.StringIO()
    six.print_('Hello,', 'person!', file=out)
    assert out.getvalue() == 'Hello, person!\n'
    out = six.StringIO()
    six.print_('Hello,', 'person!', file=out, end='')
    assert out.getvalue() == 'Hello, person!'
    out = six.StringIO()
    six.print_('Hello,', 'person!', file=out, sep='X')
    assert out.getvalue() == 'Hello,Xperson!\n'
    out = six.StringIO()
    six.print_(six.u('Hello,'), six.u('person!'), file=out)
    result = out.getvalue()
    assert isinstance(result, six.text_type)
    assert result == six.u('Hello, person!\n')
    six.print_('Hello', file=None)
    out = six.StringIO()
    six.print_(None, file=out)
    assert out.getvalue() == 'None\n'

    class FlushableStringIO(six.StringIO):

        def __init__(self):
            six.StringIO.__init__(self)
            self.flushed = False

        def flush(self):
            self.flushed = True
    out = FlushableStringIO()
    six.print_('Hello', file=out)
    assert not out.flushed
    six.print_('Hello', file=out, flush=True)
    assert out.flushed