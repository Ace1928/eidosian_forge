import html
import os
from os.path import realpath, join, dirname
import sys
import time
import warnings
import webbrowser
import jinja2
from ..frontend_semver import DECKGL_SEMVER
def deck_to_html(deck_json, mapbox_key=None, google_maps_key=None, filename=None, open_browser=False, notebook_display=None, css_background_color=None, iframe_height=500, iframe_width='100%', tooltip=True, custom_libraries=None, configuration=None, as_string=False, offline=False):
    """Converts deck.gl format JSON to an HTML page"""
    html_str = render_json_to_html(deck_json, mapbox_key=mapbox_key, google_maps_key=google_maps_key, tooltip=tooltip, css_background_color=css_background_color, custom_libraries=custom_libraries, configuration=configuration, offline=offline)
    if filename:
        with open(filename, 'w+', encoding='utf-8') as f:
            f.write(html_str)
        if open_browser:
            display_html(realpath(f.name))
    if notebook_display is None:
        notebook_display = in_jupyter()
    if in_google_colab:
        render_for_colab(html_str, iframe_height)
        return None
    elif not filename and as_string:
        return html_str
    elif notebook_display:
        return iframe_with_srcdoc(html_str, iframe_width, iframe_height)