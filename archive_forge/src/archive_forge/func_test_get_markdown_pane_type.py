import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_get_markdown_pane_type():
    assert PaneBase.get_pane_type('**Markdown**') is Markdown