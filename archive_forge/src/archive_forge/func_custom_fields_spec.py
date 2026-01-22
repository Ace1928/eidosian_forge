from __future__ import absolute_import, division, print_function
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import (
import datetime
import os
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
def custom_fields_spec():
    return dict(text1=dict(type='str'), text2=dict(type='str'), text3=dict(type='str'), text4=dict(type='str'), text5=dict(type='str'), text6=dict(type='str'), text7=dict(type='str'), text8=dict(type='str'), text9=dict(type='str'), text10=dict(type='str'), text11=dict(type='str'), text12=dict(type='str'), text13=dict(type='str'), text14=dict(type='str'), text15=dict(type='str'), number1=dict(type='float'), number2=dict(type='float'), number3=dict(type='float'), number4=dict(type='float'), number5=dict(type='float'), date1=dict(type='str'), date2=dict(type='str'), date3=dict(type='str'), date4=dict(type='str'), date5=dict(type='str'), email1=dict(type='str'), email2=dict(type='str'), email3=dict(type='str'), email4=dict(type='str'), email5=dict(type='str'), dropdown1=dict(type='str'), dropdown2=dict(type='str'), dropdown3=dict(type='str'), dropdown4=dict(type='str'), dropdown5=dict(type='str'))