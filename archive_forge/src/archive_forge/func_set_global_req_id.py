import re
from oslo_context import context
import webob.dec
from oslo_middleware import base
def set_global_req_id(self, req):
    gr_id = req.headers.get(INBOUND_HEADER, '')
    if re.match(ID_FORMAT, gr_id):
        req.environ[GLOBAL_REQ_ID] = gr_id