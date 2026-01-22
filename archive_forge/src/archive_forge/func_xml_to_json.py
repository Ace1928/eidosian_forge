from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.aci.plugins.module_utils.aci import ACIModule, aci_argument_spec, aci_annotation_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes
def xml_to_json(aci, response_data):
    """
    This function is used to convert preview XML data into JSON.
    """
    if XML_TO_JSON:
        xml = lxml.etree.fromstring(to_bytes(response_data))
        xmldata = cobra.data(xml)
        aci.result['preview'] = xmldata
    else:
        aci.result['preview'] = response_data