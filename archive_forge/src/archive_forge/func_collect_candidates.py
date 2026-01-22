import os
from glob import glob
from os import path
from typing import Any, Dict, List, Set
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import relative_path
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.i18n import get_image_filename_for_language, search_image_for_language
from sphinx.util.images import guess_mimetype
def collect_candidates(self, env: BuildEnvironment, imgpath: str, candidates: Dict[str, str], node: Node) -> None:
    globbed: Dict[str, List[str]] = {}
    for filename in glob(imgpath):
        new_imgpath = relative_path(path.join(env.srcdir, 'dummy'), filename)
        try:
            mimetype = guess_mimetype(filename)
            if mimetype is None:
                basename, suffix = path.splitext(filename)
                mimetype = 'image/x-' + suffix[1:]
            if mimetype not in candidates:
                globbed.setdefault(mimetype, []).append(new_imgpath)
        except OSError as err:
            logger.warning(__('image file %s not readable: %s') % (filename, err), location=node, type='image', subtype='not_readable')
    for key, files in globbed.items():
        candidates[key] = sorted(files, key=len)[0]