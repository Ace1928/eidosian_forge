from collections import ChainMap
from jinja2.utils import missing
from ansible.errors import AnsibleError, AnsibleUndefinedVariable
from ansible.module_utils.common.text.converters import to_native
If locals are provided, create a copy of self containing those
        locals in addition to what is already in this variable proxy.
        