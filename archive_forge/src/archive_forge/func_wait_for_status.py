import http.client as http
import time
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import requests
def wait_for_status(request_path, request_headers, status='active', max_sec=10, delay_sec=0.2, start_delay_sec=None):
    """
    Performs a time-bounded wait for the entity at the request_path to
    reach the requested status.

    :param request_path: path to use to make the request
    :param request_headers: headers to use when making the request
    :param status: the status to wait for (default: 'active')
    :param max_sec: the maximum number of seconds to wait (default: 10)
    :param delay_sec: seconds to sleep before the next request is
                      made (default: 0.2)
    :param start_delay_sec: seconds to wait before making the first
                            request (default: None)
    :raises Exception: if the entity fails to reach the status within
                       the requested time or if the server returns something
                       other than a 200 response
    """
    start_time = time.time()
    done_time = start_time + max_sec
    if start_delay_sec:
        time.sleep(start_delay_sec)
    while time.time() <= done_time:
        resp = requests.get(request_path, headers=request_headers)
        if resp.status_code != http.OK:
            raise Exception('Received {} response from server'.format(resp.status_code))
        entity = jsonutils.loads(resp.text)
        if entity['status'] == status:
            return
        time.sleep(delay_sec)
    entity_id = request_path.rsplit('/', 1)[1]
    msg = "Entity {0} failed to reach status '{1}' within {2} sec"
    raise Exception(msg.format(entity_id, status, max_sec))