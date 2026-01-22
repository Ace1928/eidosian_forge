import os.path as op
from uuid import uuid1
import time
from pyxnat.tests import skip_if_no_network
from pyxnat import Interface
from pyxnat.core import interfaces
@skip_if_no_network
def test_06_list_project_attrs():
    project_attributes = ['xnat:projectData/name', 'xnat:projectData/type', 'xnat:projectData/description', 'xnat:projectData/keywords', 'xnat:projectData/aliases', 'xnat:projectData/aliases/alias', 'xnat:projectData/aliases/alias/None', 'xnat:projectData/publications', 'xnat:projectData/publications/publication', 'xnat:projectData/resources', 'xnat:projectData/resources/resource', 'xnat:projectData/studyProtocol', 'xnat:projectData/PI', 'xnat:projectData/investigators', 'xnat:projectData/investigators/investigator', 'xnat:projectData/fields', 'xnat:projectData/fields/field', 'xnat:projectData/fields/field/None']
    p = central.select.project('pyxnat_tests')
    assert p.attrs() == []
    central.manage.schemas.add('xapi/schemas/xnat')
    p = central.select.project('pyxnat_tests')
    assert p.attrs() == project_attributes