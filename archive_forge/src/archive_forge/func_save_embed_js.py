import os
import io
import sys
import zipfile
import shutil
from ipywidgets import embed as wembed
import ipyvolume
from ipyvolume.utils import download_to_file, download_to_bytes
from ipyvolume._version import __version_threejs__
def save_embed_js(target='', version=wembed.__html_manager_version__):
    """Download and save the ipywidgets embedding javascript to a local file.

    :type target: str
    :type version: str

    """
    url = u'https://unpkg.com/@jupyter-widgets/html-manager@{0:s}/dist/embed-amd.js'.format(version)
    if version.startswith('^'):
        version = version[1:]
    filename = 'embed-amd_v{0:s}.js'.format(version)
    filepath = os.path.join(target, filename)
    download_to_file(url, filepath)
    return filename