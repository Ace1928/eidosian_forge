import numpy as np
import pandas._config.config as cf
from pandas import (
def test_enable_data_resource_formatter(self, ip):
    formatters = ip.instance(config=ip.config).display_formatter.formatters
    mimetype = 'application/vnd.dataresource+json'
    with cf.option_context('display.html.table_schema', True):
        assert 'application/vnd.dataresource+json' in formatters
        assert formatters[mimetype].enabled
    assert 'application/vnd.dataresource+json' in formatters
    assert not formatters[mimetype].enabled
    with cf.option_context('display.html.table_schema', True):
        assert 'application/vnd.dataresource+json' in formatters
        assert formatters[mimetype].enabled
        ip.instance(config=ip.config).display_formatter.format(cf)