import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_markdown_pane_markdown_it_renderer(document, comm):
    pane = Markdown('\n    - [x] Task1\n    - [ ] Task2\n    ', renderer='markdown-it')
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text == '&lt;ul class=&quot;contains-task-list&quot;&gt;\n&lt;li class=&quot;task-list-item&quot;&gt;&lt;input class=&quot;task-list-item-checkbox&quot; checked=&quot;checked&quot; disabled=&quot;disabled&quot; type=&quot;checkbox&quot;&gt; Task1&lt;/li&gt;\n&lt;li class=&quot;task-list-item&quot;&gt;&lt;input class=&quot;task-list-item-checkbox&quot; disabled=&quot;disabled&quot; type=&quot;checkbox&quot;&gt; Task2&lt;/li&gt;\n&lt;/ul&gt;\n'