import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def fake_auth_request_v2(*args, **kwargs):
    s_url = 'http://127.0.0.1:8080/v1.0/AUTH_fakeuser'
    resp = {'access': {'token': {'id': '12' * 10}, 'serviceCatalog': [{'type': 'object-store', 'endpoints': [{'region': 'test', 'internalURL': s_url}]}]}}
    ret = Response(status=200, content=json.dumps(resp))
    return ret