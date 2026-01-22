from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote_plus, urlparse
from ..core.templates import AUTOLOAD_REQUEST_TAG, FILE
from ..resources import DEFAULT_SERVER_HTTP_URL
from ..util.serialization import make_globally_unique_css_safe_id
from ..util.strings import format_docstring
from .bundle import bundle_for_objs_and_resources
from .elements import html_page_for_render_items
from .util import RenderItem
def server_html_page_for_session(session: ServerSession, resources: Resources, title: str, template: Template=FILE, template_variables: dict[str, Any] | None=None):
    """

    Args:
        session (ServerSession) :

        resources (Resources) :

        title (str) :

        template (Template) :

        template_variables (dict) :

    Returns:
        str

    """
    render_item = RenderItem(token=session.token, roots=session.document.roots, use_for_title=True)
    if template_variables is None:
        template_variables = {}
    bundle = bundle_for_objs_and_resources(None, resources)
    html = html_page_for_render_items(bundle, {}, [render_item], title, template=template, template_variables=template_variables)
    return html