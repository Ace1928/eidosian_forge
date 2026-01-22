import numpy as np
import pandas._config.config as cf
from pandas import (
def test_publishes(self, ip):
    ipython = ip.instance(config=ip.config)
    df = DataFrame({'A': [1, 2]})
    objects = [df['A'], df]
    expected_keys = [{'text/plain', 'application/vnd.dataresource+json'}, {'text/plain', 'text/html', 'application/vnd.dataresource+json'}]
    opt = cf.option_context('display.html.table_schema', True)
    last_obj = None
    for obj, expected in zip(objects, expected_keys):
        last_obj = obj
        with opt:
            formatted = ipython.display_formatter.format(obj)
        assert set(formatted[0].keys()) == expected
    with_latex = cf.option_context('styler.render.repr', 'latex')
    with opt, with_latex:
        formatted = ipython.display_formatter.format(last_obj)
    expected = {'text/plain', 'text/html', 'text/latex', 'application/vnd.dataresource+json'}
    assert set(formatted[0].keys()) == expected