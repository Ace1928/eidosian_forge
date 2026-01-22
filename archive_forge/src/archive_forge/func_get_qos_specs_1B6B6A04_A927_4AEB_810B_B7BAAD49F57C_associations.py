from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C_associations(self, **kw):
    type_id1 = '4230B13A-7A37-4E84-B777-EFBA6FCEE4FF'
    type_id2 = '4230B13A-AB37-4E84-B777-EFBA6FCEE4FF'
    type_name1 = 'type1'
    type_name2 = 'type2'
    return (202, {}, {'qos_associations': [_stub_qos_associates(type_id1, type_name1), _stub_qos_associates(type_id2, type_name2)]})