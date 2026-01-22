import plistlib
import shlex
import subprocess
from os import environ
from os import path
from subprocess import CalledProcessError, PIPE, STDOUT
from typing import Any
import sphinx
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.errors import SphinxError
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset, copy_asset_file
from sphinx.util.matching import Matcher
from sphinx.util.osutil import ensuredir, make_filename
@progress_message(__('generating help index'))
def build_helpindex(self, language_dir: str) -> None:
    """Generate the help index."""
    args = [self.config.applehelp_indexer_path, '-Cf', path.join(language_dir, 'search.helpindex'), language_dir]
    if self.config.applehelp_index_anchors is not None:
        args.append('-a')
    if self.config.applehelp_min_term_length is not None:
        args += ['-m', '%s' % self.config.applehelp_min_term_length]
    if self.config.applehelp_stopwords is not None:
        args += ['-s', self.config.applehelp_stopwords]
    if self.config.applehelp_locale is not None:
        args += ['-l', self.config.applehelp_locale]
    if self.config.applehelp_disable_external_tools:
        raise SkipProgressMessage(__('you will need to index this help book with:\n  %s'), ' '.join([shlex.quote(arg) for arg in args]))
    else:
        try:
            subprocess.run(args, stdout=PIPE, stderr=STDOUT, check=True)
        except OSError:
            raise AppleHelpIndexerFailed(__('Command not found: %s') % args[0])
        except CalledProcessError as exc:
            raise AppleHelpIndexerFailed(exc.stdout)