from uuid import uuid1
from pyxnat import Interface
from pyxnat import jsonutil
import os.path as op
def test_delete_search_template():
    central.manage.search.delete_template(search_template_name)
    assert search_template_name not in central.manage.search.saved_templates()