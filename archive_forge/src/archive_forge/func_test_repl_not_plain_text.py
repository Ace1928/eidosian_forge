import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory
def test_repl_not_plain_text(self):
    ip = get_ipython()
    formatter = ip.display_formatter
    assert formatter.active_types == ['text/plain']
    assert formatter.ipython_display_formatter.enabled

    class Test(object):

        def __repr__(self):
            return '<Test %i>' % id(self)

        def _repr_html_(self):
            return '<html>'
    obj = Test()
    data, _ = formatter.format(obj)
    self.assertEqual(data, {'text/plain': repr(obj)})

    class Test2(Test):

        def _ipython_display_(self):
            from IPython.display import display, HTML
            display(HTML('<custom>'))
    called = False

    def handler(data, metadata):
        print('Handler called')
        nonlocal called
        called = True
    ip.display_formatter.active_types.append('text/html')
    ip.display_formatter.formatters['text/html'].enabled = True
    ip.mime_renderers['text/html'] = handler
    try:
        obj = Test()
        display(obj)
    finally:
        ip.display_formatter.formatters['text/html'].enabled = False
        del ip.mime_renderers['text/html']
    assert called == True