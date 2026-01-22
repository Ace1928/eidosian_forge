import json
import urllib
import numpy as np
import pytest
import cirq
from cirq import quirk_url_to_circuit, quirk_json_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
@pytest.mark.parametrize('url,msg', [('http://algassert.com/quirk#circuit=[]', 'top-level dictionary'), ('http://algassert.com/quirk#circuit={}', '"cols" entry'), ('http://algassert.com/quirk#circuit={"cols": 1}', 'cols must be a list'), ('http://algassert.com/quirk#circuit={"cols": [0]}', 'col must be a list'), ('http://algassert.com/quirk#circuit={"cols": [[0]]}', 'Unrecognized column entry: 0'), ('http://algassert.com/quirk#circuit={"cols": [["not a real"]]}', 'Unrecognized column entry: '), ('http://algassert.com/quirk#circuit={"cols": [[]], "other": 1}', 'Unrecognized Circuit JSON keys')])
def test_parse_failures(url, msg):
    parsed_url = urllib.parse.urlparse(url)
    data = json.loads(parsed_url.fragment[len('circuit='):])
    with pytest.raises(ValueError, match=msg):
        _ = quirk_url_to_circuit(url)
    with pytest.raises(ValueError, match=msg):
        _ = quirk_json_to_circuit(data)