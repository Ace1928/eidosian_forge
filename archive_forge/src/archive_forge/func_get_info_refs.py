import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
def get_info_refs(req, backend, mat):
    params = parse_qs(req.environ['QUERY_STRING'])
    service = params.get('service', [None])[0]
    try:
        repo = get_repo(backend, mat)
    except NotGitRepository as e:
        yield req.not_found(str(e))
        return
    if service and (not req.dumb):
        handler_cls = req.handlers.get(service.encode('ascii'), None)
        if handler_cls is None:
            yield req.forbidden('Unsupported service')
            return
        req.nocache()
        write = req.respond(HTTP_OK, 'application/x-%s-advertisement' % service)
        proto = ReceivableProtocol(BytesIO().read, write)
        handler = handler_cls(backend, [url_prefix(mat)], proto, stateless_rpc=True, advertise_refs=True)
        handler.proto.write_pkt_line(b'# service=' + service.encode('ascii') + b'\n')
        handler.proto.write_pkt_line(None)
        handler.handle()
    else:
        req.nocache()
        req.respond(HTTP_OK, 'text/plain')
        logger.info('Emulating dumb info/refs')
        yield from generate_info_refs(repo)