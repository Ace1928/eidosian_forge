import os
import json
from shutil import copyfile, rmtree
from docker.tls import TLSConfig
from docker.errors import ContextException
from docker.context.config import get_meta_dir
from docker.context.config import get_meta_file
from docker.context.config import get_tls_dir
from docker.context.config import get_context_host
def set_endpoint(self, name='docker', host=None, tls_cfg=None, skip_tls_verify=False, def_namespace=None):
    self.endpoints[name] = {'Host': get_context_host(host, not skip_tls_verify), 'SkipTLSVerify': skip_tls_verify}
    if def_namespace:
        self.endpoints[name]['DefaultNamespace'] = def_namespace
    if tls_cfg:
        self.tls_cfg[name] = tls_cfg