from __future__ import absolute_import, division, print_function
import ast
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.netconf.netconf import (
def get_xml_request(module, request, xmlns, content):
    if content is None:
        if xmlns is None:
            return '<%s/>' % request
        else:
            return '<%s xmlns="%s"/>' % (request, xmlns)
    if isinstance(content, str):
        content = content.strip()
        if content.startswith('<') and content.endswith('>'):
            if xmlns is None:
                return '<%s>%s</%s>' % (request, content, request)
            else:
                return '<%s xmlns="%s">%s</%s>' % (request, xmlns, content, request)
        try:
            content = ast.literal_eval(content)
        except Exception:
            module.fail_json(msg='unsupported content value `%s`' % content)
    if isinstance(content, dict):
        if not HAS_JXMLEASE:
            module.fail_json(msg='jxmlease is required to convert RPC content to XML but does not appear to be installed. It can be installed using `pip install jxmlease`')
        payload = jxmlease.XMLDictNode(content).emit_xml(pretty=False, full_document=False)
        if xmlns is None:
            return '<%s>%s</%s>' % (request, payload, request)
        else:
            return '<%s xmlns="%s">%s</%s>' % (request, xmlns, payload, request)
    module.fail_json(msg='unsupported content data-type `%s`' % type(content).__name__)