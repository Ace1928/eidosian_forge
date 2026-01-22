from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import fetch_url
def axapi_call(module, url, post=None):
    """
    Returns a datastructure based on the result of the API call
    """
    rsp, info = fetch_url(module, url, data=post)
    if not rsp or info['status'] >= 400:
        module.fail_json(msg='failed to connect (status code %s), error was %s' % (info['status'], info.get('msg', 'no error given')))
    try:
        raw_data = rsp.read()
        data = json.loads(raw_data)
    except ValueError:
        if 'status="ok"' in raw_data.lower():
            data = {'response': {'status': 'OK'}}
        else:
            data = {'response': {'status': 'fail', 'err': {'msg': raw_data}}}
    except Exception:
        module.fail_json(msg='could not read the result from the host')
    finally:
        rsp.close()
    return data