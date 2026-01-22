import html
import os
from os.path import realpath, join, dirname
import sys
import time
import warnings
import webbrowser
import jinja2
from ..frontend_semver import DECKGL_SEMVER
def cdn_picker(offline=False):
    dev_port = os.getenv('PYDECK_DEV_PORT')
    if dev_port:
        print('pydeck running in development mode, expecting @deck.gl/jupyter-widget served at {}'.format(dev_port))
        return "<script type='text/javascript' src='http://localhost:{dev_port}/dist/index.js'></script>\n<script type='text/javascript' src='http://localhost:{dev_port}/dist/index.js.map'></script>\n".format(dev_port=dev_port)
    if offline:
        RELPATH_TO_BUNDLE = '../nbextension/static/index.js'
        with open(join(dirname(__file__), RELPATH_TO_BUNDLE), 'r', encoding='utf-8') as file:
            js = file.read()
        return "<script type='text/javascript'>{}</script>".format(js)
    return "<script src='{}'></script>".format(CDN_URL)