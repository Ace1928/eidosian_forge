from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote_plus
from ..core.types import ID
from ..document import Document
from ..resources import DEFAULT_SERVER_HTTP_URL, SessionCoordinates
from ..util.browser import NEW_PARAM, BrowserLike, BrowserTarget
from ..util.token import generate_jwt_token, generate_session_id
from .states import ErrorReason
from .util import server_url_for_websocket_url, websocket_url_for_server_url
def show_session(session_id: ID | None=None, url: str='default', session: ClientSession | None=None, browser: str | None=None, new: BrowserTarget='tab', controller: BrowserLike | None=None) -> None:
    """ Open a browser displaying a session document.

        If you have a session from ``pull_session()`` or ``push_session`` you
        can ``show_session(session=mysession)``. If you don't need to open a
        connection to the server yourself, you can show a new session in a
        browser by providing just the ``url``.

        Args:
            session_id (string, optional) :
               The name of the session, None to autogenerate a random one (default: None)

            url : (str, optional): The URL to a Bokeh application on a Bokeh server
                can also be `"default"` which will connect to the default app URL

            session (ClientSession, optional) : session to get session ID and server URL from
                If you specify this, you don't need to specify session_id and url

            browser (str, optional) : browser to show with (default: None)
                For systems that support it, the **browser** argument allows
                specifying which browser to display in, e.g. "safari", "firefox",
                "opera", "windows-default" (see the :doc:`webbrowser <python:library/webbrowser>`
                module documentation in the standard lib for more details).

            new (str, optional) : new file output mode (default: "tab")
                For file-based output, opens or raises the browser window
                showing the current output file.  If **new** is 'tab', then
                opens a new tab. If **new** is 'window', then opens a new window.

        """
    if session is not None:
        server_url = server_url_for_websocket_url(session._connection.url)
        session_id = session.id
    else:
        coords = SessionCoordinates(session_id=session_id, url=url)
        server_url = coords.url
        session_id = coords.session_id
    if controller is None:
        from bokeh.util.browser import get_browser_controller
        controller = get_browser_controller(browser=browser)
    controller.open(server_url + '?bokeh-session-id=' + quote_plus(session_id), new=NEW_PARAM[new])