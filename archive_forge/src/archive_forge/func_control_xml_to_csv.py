from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
import traceback
import sys
import os
def control_xml_to_csv(filepath, module):
    if not HAS_BS4_LIBRARY:
        module.fail_json(msg=missing_required_lib('bs4'), exception=BS4_LIBRARY_IMPORT_ERROR)
    infile = open(filepath + '/control_utf8.xml', 'r')
    contents = infile.read()
    soup = BeautifulSoup(markup=contents, features='lxml-xml')
    space = soup.find('components')
    component_list = space.findChildren('component', recursive=False)
    csv_output = open('control_output.csv', 'w')
    csv_header = '"' + 'Component Name' + '","' + 'Component Display Name' + '","' + 'Parameter Name' + '","' + 'Parameter Inifile Key' + '","' + 'Parameter Access' + '","' + 'Parameter Encode' + '","' + 'Parameter Default Value' + '","' + 'Parameter Inifile description' + '"'
    csv_output.write('%s\n' % csv_header)
    for component in component_list:
        for parameter in component.findChildren('parameter'):
            component_key = parameter.findParent('component')
            component_key_name_text = component_key['name']
            for child in component_key.findChildren('display-name'):
                component_key_display_name_text = child.get_text().replace('\n', '')
            component_parameter_key_name = parameter['name']
            component_parameter_key_inifile_name = parameter.get('defval-for-inifile-generation', '')
            component_parameter_key_access = parameter.get('access', '')
            component_parameter_key_encode = parameter.get('encode', '')
            component_parameter_key_defval = parameter.get('defval', '')
            component_parameter_contents_doclong_text = parameter.get_text().replace('\n', '')
            component_parameter_contents_doclong_text_quote_replacement = component_parameter_contents_doclong_text.replace('"', "'")
            csv_string = '"' + component_key_name_text + '","' + component_key_display_name_text + '","' + component_parameter_key_name + '","' + component_parameter_key_inifile_name + '","' + component_parameter_key_access + '","' + component_parameter_key_encode + '","' + component_parameter_key_defval + '","' + component_parameter_contents_doclong_text_quote_replacement + '"'
            csv_output.write('%s\n' % csv_string)
    csv_output.close()