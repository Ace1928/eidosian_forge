import re
import threading
def make_prefix_middleware(app, global_conf, prefix='/', translate_forwarded_server=True, force_port=None, scheme=None):
    from paste.deploy.converters import asbool
    translate_forwarded_server = asbool(translate_forwarded_server)
    return PrefixMiddleware(app, prefix=prefix, translate_forwarded_server=translate_forwarded_server, force_port=force_port, scheme=scheme)