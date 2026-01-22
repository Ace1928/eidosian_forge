from typing import Any, Dict, Set
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.transforms import SphinxContentsFilter
Add a title node to the document (just copy the first section title),
        and store that title in the environment.
        