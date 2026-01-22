from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
Convert data which is in json to xml"

    :param data: The data passed in (data|to_xml(...))
    :type data: xml
    :param engine: Conversion library default=xmltodict
    :param indent: Indent char default='tabs'
    :param indent_width: Indent char multiplier default=4
    :param full_document: Flag to disable xml declaration
    