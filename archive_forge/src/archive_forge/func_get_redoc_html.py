import json
from typing import Any, Dict, Optional
from fastapi.encoders import jsonable_encoder
from starlette.responses import HTMLResponse
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]
def get_redoc_html(*, openapi_url: Annotated[str, Doc('\n            The OpenAPI URL that ReDoc should load and use.\n\n            This is normally done automatically by FastAPI using the default URL\n            `/openapi.json`.\n            ')], title: Annotated[str, Doc('\n            The HTML `<title>` content, normally shown in the browser tab.\n            ')], redoc_js_url: Annotated[str, Doc('\n            The URL to use to load the ReDoc JavaScript.\n\n            It is normally set to a CDN URL.\n            ')]='https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js', redoc_favicon_url: Annotated[str, Doc('\n            The URL of the favicon to use. It is normally shown in the browser tab.\n            ')]='https://fastapi.tiangolo.com/img/favicon.png', with_google_fonts: Annotated[bool, Doc('\n            Load and use Google Fonts.\n            ')]=True) -> HTMLResponse:
    """
    Generate and return the HTML response that loads ReDoc for the alternative
    API docs (normally served at `/redoc`).

    You would only call this function yourself if you needed to override some parts,
    for example the URLs to use to load ReDoc's JavaScript and CSS.

    Read more about it in the
    [FastAPI docs for Custom Docs UI Static Assets (Self-Hosting)](https://fastapi.tiangolo.com/how-to/custom-docs-ui-assets/).
    """
    html = f'\n    <!DOCTYPE html>\n    <html>\n    <head>\n    <title>{title}</title>\n    <!-- needed for adaptive design -->\n    <meta charset="utf-8"/>\n    <meta name="viewport" content="width=device-width, initial-scale=1">\n    '
    if with_google_fonts:
        html += '\n    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">\n    '
    html += f'''\n    <link rel="shortcut icon" href="{redoc_favicon_url}">\n    <!--\n    ReDoc doesn't change outer page styles\n    -->\n    <style>\n      body {{\n        margin: 0;\n        padding: 0;\n      }}\n    </style>\n    </head>\n    <body>\n    <noscript>\n        ReDoc requires Javascript to function. Please enable it to browse the documentation.\n    </noscript>\n    <redoc spec-url="{openapi_url}"></redoc>\n    <script src="{redoc_js_url}"> </script>\n    </body>\n    </html>\n    '''
    return HTMLResponse(html)