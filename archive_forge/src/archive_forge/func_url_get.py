from __future__ import absolute_import, division, print_function
import os
import re
import shutil
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.module_utils.compat.datetime import utcnow, utcfromtimestamp
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url, url_argument_spec
def url_get(module, url, dest, use_proxy, last_mod_time, force, timeout=10, headers=None, tmp_dest='', method='GET', unredirected_headers=None, decompress=True, ciphers=None, use_netrc=True):
    """
    Download data from the url and store in a temporary file.

    Return (tempfile, info about the request)
    """
    start = utcnow()
    rsp, info = fetch_url(module, url, use_proxy=use_proxy, force=force, last_mod_time=last_mod_time, timeout=timeout, headers=headers, method=method, unredirected_headers=unredirected_headers, decompress=decompress, ciphers=ciphers, use_netrc=use_netrc)
    elapsed = (utcnow() - start).seconds
    if info['status'] == 304:
        module.exit_json(url=url, dest=dest, changed=False, msg=info.get('msg', ''), status_code=info['status'], elapsed=elapsed)
    if info['status'] == -1:
        module.fail_json(msg=info['msg'], url=url, dest=dest, elapsed=elapsed)
    if info['status'] != 200 and (not url.startswith('file:/')) and (not (url.startswith('ftp:/') and info.get('msg', '').startswith('OK'))):
        module.fail_json(msg='Request failed', status_code=info['status'], response=info['msg'], url=url, dest=dest, elapsed=elapsed)
    if tmp_dest:
        tmp_dest_is_dir = os.path.isdir(tmp_dest)
        if not tmp_dest_is_dir:
            if os.path.exists(tmp_dest):
                module.fail_json(msg='%s is a file but should be a directory.' % tmp_dest, elapsed=elapsed)
            else:
                module.fail_json(msg='%s directory does not exist.' % tmp_dest, elapsed=elapsed)
    else:
        tmp_dest = module.tmpdir
    fd, tempname = tempfile.mkstemp(dir=tmp_dest)
    f = os.fdopen(fd, 'wb')
    try:
        shutil.copyfileobj(rsp, f)
    except Exception as e:
        os.remove(tempname)
        module.fail_json(msg='failed to create temporary content file: %s' % to_native(e), elapsed=elapsed, exception=traceback.format_exc())
    f.close()
    rsp.close()
    return (tempname, info)