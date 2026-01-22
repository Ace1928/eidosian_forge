from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page
def continue_with_auth(request_id: RequestId, auth_challenge_response: AuthChallengeResponse) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Continues a request supplying authChallengeResponse following authRequired event.

    :param request_id: An id the client received in authRequired event.
    :param auth_challenge_response: Response to  with an authChallenge.
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['authChallengeResponse'] = auth_challenge_response.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Fetch.continueWithAuth', 'params': params}
    json = (yield cmd_dict)