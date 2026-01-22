from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import quote
import json
import re
import xml.etree.ElementTree as ET
This is basically as follows:
        <filtered-models>
            <and>
                <equals>
                    <attribute id=...>
                        <value>...</value>
                    </attribute>
                </equals>
                <equals>
                    <attribute...>
                </equals>
            </and>
        </filtered-models>
        