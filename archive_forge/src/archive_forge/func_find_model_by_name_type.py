from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import quote
import json
import re
import xml.etree.ElementTree as ET
def find_model_by_name_type(self, mname, mtype, ret_attrs=None):
    """
        Find a model by name and type
        :param mname: Model name
        :type mname: str
        :param mtype: Model type
        :type mtype: str
        :param ret_attrs: List of attributes by name or ID to return back
            (default is Model_Handle)
        :type ret_attrs: list
        returns: find_model(): Dictionary mapping of ret_attrs to values:
            {ret_attr: ret_val}
        rtype: dict
        """
    if ret_attrs is None:
        ret_attrs = ['Model_Handle']
    'This is basically as follows:\n        <filtered-models>\n            <and>\n                <equals>\n                    <attribute id=...>\n                        <value>...</value>\n                    </attribute>\n                </equals>\n                <equals>\n                    <attribute...>\n                </equals>\n            </and>\n        </filtered-models>\n        '
    filtered_models = ET.Element('filtered-models')
    _and = ET.SubElement(filtered_models, 'and')
    MN_equals = ET.SubElement(_and, 'equals')
    Model_Name = ET.SubElement(MN_equals, 'attribute', {'id': self.attr_map['Model_Name']})
    MN_value = ET.SubElement(Model_Name, 'value')
    MN_value.text = mname
    MTN_equals = ET.SubElement(_and, 'equals')
    Modeltype_Name = ET.SubElement(MTN_equals, 'attribute', {'id': self.attr_map['Modeltype_Name']})
    MTN_value = ET.SubElement(Modeltype_Name, 'value')
    MTN_value.text = mtype
    return self.find_model(ET.tostring(filtered_models, encoding='unicode'), ret_attrs)