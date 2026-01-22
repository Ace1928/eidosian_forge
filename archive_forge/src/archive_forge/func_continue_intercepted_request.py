from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
def continue_intercepted_request(interception_id: InterceptionId, error_reason: typing.Optional[ErrorReason]=None, raw_response: typing.Optional[str]=None, url: typing.Optional[str]=None, method: typing.Optional[str]=None, post_data: typing.Optional[str]=None, headers: typing.Optional[Headers]=None, auth_challenge_response: typing.Optional[AuthChallengeResponse]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Response to Network.requestIntercepted which either modifies the request to continue with any
    modifications, or blocks it, or completes it with the provided response bytes. If a network
    fetch occurs as a result which encounters a redirect an additional Network.requestIntercepted
    event will be sent with the same InterceptionId.
    Deprecated, use Fetch.continueRequest, Fetch.fulfillRequest and Fetch.failRequest instead.

    **EXPERIMENTAL**

    :param interception_id:
    :param error_reason: *(Optional)* If set this causes the request to fail with the given reason. Passing ```Aborted```` for requests marked with ````isNavigationRequest``` also cancels the navigation. Must not be set in response to an authChallenge.
    :param raw_response: *(Optional)* If set the requests completes using with the provided base64 encoded raw response, including HTTP status line and headers etc... Must not be set in response to an authChallenge.
    :param url: *(Optional)* If set the request url will be modified in a way that's not observable by page. Must not be set in response to an authChallenge.
    :param method: *(Optional)* If set this allows the request method to be overridden. Must not be set in response to an authChallenge.
    :param post_data: *(Optional)* If set this allows postData to be set. Must not be set in response to an authChallenge.
    :param headers: *(Optional)* If set this allows the request headers to be changed. Must not be set in response to an authChallenge.
    :param auth_challenge_response: *(Optional)* Response to a requestIntercepted with an authChallenge. Must not be set otherwise.
    """
    params: T_JSON_DICT = dict()
    params['interceptionId'] = interception_id.to_json()
    if error_reason is not None:
        params['errorReason'] = error_reason.to_json()
    if raw_response is not None:
        params['rawResponse'] = raw_response
    if url is not None:
        params['url'] = url
    if method is not None:
        params['method'] = method
    if post_data is not None:
        params['postData'] = post_data
    if headers is not None:
        params['headers'] = headers.to_json()
    if auth_challenge_response is not None:
        params['authChallengeResponse'] = auth_challenge_response.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.continueInterceptedRequest', 'params': params}
    json = (yield cmd_dict)