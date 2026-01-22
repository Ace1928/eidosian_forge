import os
import urllib
from typing import Any, Dict
import sphinx
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
def create_nojekyll_and_cname(app: Sphinx, env: BuildEnvironment) -> None:
    if app.builder.format == 'html':
        open(os.path.join(app.builder.outdir, '.nojekyll'), 'wb').close()
        html_baseurl = app.config.html_baseurl
        if html_baseurl:
            domain = urllib.parse.urlparse(html_baseurl).hostname
            if domain and (not domain.endswith('.github.io')):
                with open(os.path.join(app.builder.outdir, 'CNAME'), 'w', encoding='utf-8') as f:
                    f.write(domain)