from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
import traceback
import sys
import os
def product_catalog_xml_to_csv(filepath, module):
    if not HAS_BS4_LIBRARY:
        module.fail_json(msg=missing_required_lib('bs4'), exception=BS4_LIBRARY_IMPORT_ERROR)
    infile = open(filepath + '/product_catalog_utf8.xml', 'r')
    contents = infile.read()
    soup = BeautifulSoup(markup=contents, features='lxml-xml')
    space = soup.find_all('component')
    csv_output = open('product_catalog_output.csv', 'w')
    csv_header = '"' + 'Product Catalog Component Name' + '","' + 'Product Catalog Component ID' + '","' + 'Product Catalog Component Table' + '","' + 'Product Catalog Component Output Dir' + '","' + 'Product Catalog Component Display Name' + '","' + 'Product Catalog Component UserInfo' + '"'
    csv_output.write('%s\n' % csv_header)
    for component in space:
        component_name = component.get('name', '')
        component_id = component.get('id', '')
        component_table = component.get('table', '')
        component_output_dir = component.get('output-dir', '')
        for displayname in component.findChildren('display-name'):
            component_displayname = displayname.get_text().strip()
        for userinfo in component.findChildren('user-info'):
            html_raw = userinfo.get_text().strip()
            html_parsed = BeautifulSoup(html_raw, 'html.parser')
            component_userinfo = html_parsed.get_text().replace('"', "'")
        csv_string = '"' + component_name + '","' + component_id + '","' + component_table + '","' + component_output_dir + '","' + component_displayname + '","' + component_userinfo + '"'
        csv_output.write('%s\n' % csv_string)
    csv_output.close()