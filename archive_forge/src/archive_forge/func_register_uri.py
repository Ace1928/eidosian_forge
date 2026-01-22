import urllib.parse
import weakref
from requests.adapters import BaseAdapter
from requests.utils import requote_uri
from requests_mock import exceptions
from requests_mock.request import _RequestObjectProxy
from requests_mock.response import _MatcherResponse
import logging
def register_uri(self, method, url, response_list=None, **kwargs):
    """Register a new URI match and fake response.

        :param str method: The HTTP method to match.
        :param str url: The URL to match.
        """
    complete_qs = kwargs.pop('complete_qs', False)
    additional_matcher = kwargs.pop('additional_matcher', None)
    request_headers = kwargs.pop('request_headers', {})
    real_http = kwargs.pop('_real_http', False)
    json_encoder = kwargs.pop('json_encoder', None)
    if response_list and kwargs:
        raise RuntimeError('You should specify either a list of responses OR response kwargs. Not both.')
    elif real_http and (response_list or kwargs):
        raise RuntimeError('You should specify either response data OR real_http. Not both.')
    elif not response_list:
        if json_encoder is not None:
            kwargs['json_encoder'] = json_encoder
        response_list = [] if real_http else [kwargs]
    responses = [_MatcherResponse(**k) for k in response_list]
    matcher = _Matcher(method, url, responses, case_sensitive=self._case_sensitive, complete_qs=complete_qs, additional_matcher=additional_matcher, request_headers=request_headers, real_http=real_http)
    self.add_matcher(matcher)
    return matcher