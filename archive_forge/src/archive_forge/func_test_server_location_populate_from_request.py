import param
import pytest
from panel.io.location import Location
from panel.io.state import state
from panel.tests.util import serve_and_request, wait_until
from panel.util import edit_readonly
def test_server_location_populate_from_request():
    locs = []

    def app():
        loc = state.location
        locs.append(loc)
        return '# Location Test'
    request = serve_and_request(app, suffix='?foo=1')
    wait_until(lambda: len(locs) == 1)
    loc = locs[0]
    assert loc.href == request.url
    assert loc.protocol == 'http:'
    assert loc.hostname == 'localhost'
    assert loc.pathname == '/'
    assert loc.search == '?foo=1'