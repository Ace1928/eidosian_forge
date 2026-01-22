import os.path as op
from uuid import uuid1
import time
from pyxnat.tests import skip_if_no_network
from pyxnat import Interface
from pyxnat.core import interfaces
@skip_if_no_network
def test_01_fancy_resource_create():
    field_data = {'experiment': 'xnat:mrSessionData', 'ID': 'TEST_%s' % eid, 'xnat:mrSessionData/age': '42', 'xnat:subjectData/investigator/lastname': 'doe', 'xnat:subjectData/investigator/firstname': 'john', 'xnat:subjectData/ID': 'TEST_%s' % sid}
    experiment.create(**field_data)
    assert subject.exists()
    assert experiment.exists()
    globals()['subject'] = experiment.parent()
    globals()['experiment'] = experiment