import os.path
import logging
from wsme.utils import OrderedDict
from wsme.protocol import CallContext, Protocol, media_type_accept
import wsme.rest
from wsme.rest import json
from wsme.rest import xml
import wsme.runtime
def iter_calls(self, request):
    context = CallContext(request)
    context.outformat = None
    ext = os.path.splitext(request.path.split('/')[-1])[1]
    inmime = request.content_type
    try:
        offers = request.accept.acceptable_offers(self.content_types)
        outmime = offers[0][0]
    except IndexError:
        outmime = None
    outformat = None
    informat = None
    for dfname, df in self.dataformats.items():
        if ext == '.' + dfname:
            outformat = df
            if not inmime:
                informat = df
    if outformat is None and request.accept:
        for dfname, df in self.dataformats.items():
            if outmime in df.accept_content_types:
                outformat = df
                if not inmime:
                    informat = df
    if outformat is None:
        for dfname, df in self.dataformats.items():
            if inmime == df.content_type:
                outformat = df
    context.outformat = outformat
    context.outformat_options = {'nest_result': getattr(self, 'nest_result', False)}
    if not inmime and informat:
        inmime = informat.content_type
        log.debug('Inferred input type: %s' % inmime)
    context.inmime = inmime
    yield context