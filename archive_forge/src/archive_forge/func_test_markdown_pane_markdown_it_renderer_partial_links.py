import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_markdown_pane_markdown_it_renderer_partial_links(document, comm):
    pane = Markdown('[Test](http:/', renderer='markdown-it')
    model = pane.get_root(document, comm=comm)
    assert model.text == '&lt;p&gt;[Test](http:/&lt;/p&gt;\n'
    pane.object = '[Test](http://'
    assert model.text == '&lt;p&gt;[Test](http://&lt;/p&gt;\n'
    pane.object = '[Test](http://google.com)'
    assert model.text == '&lt;p&gt;&lt;a href=&quot;http://google.com&quot;&gt;Test&lt;/a&gt;&lt;/p&gt;\n'