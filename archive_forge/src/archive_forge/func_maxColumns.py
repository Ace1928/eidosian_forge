import os
import pytest
from lxml import etree
import json
import tempfile
@pytest.fixture(params=[None, 15], ids='maxColumns={}'.format)
def maxColumns(request):
    return request.param