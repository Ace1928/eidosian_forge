from uuid import uuid1
from pyxnat import Interface
from pyxnat import jsonutil
import os.path as op
def test_get_search():
    results = central.manage.search.get(search_name)
    assert isinstance(results, jsonutil.JsonTable)