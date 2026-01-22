from __future__ import annotations
import argparse
import logging
import os.path
from typing import Any
from flake8.discover_files import expand_paths
from flake8.formatting import base as formatter
from flake8.main import application as app
from flake8.options.parse_args import parse_args
def get_style_guide(**kwargs: Any) -> StyleGuide:
    """Provision a StyleGuide for use.

    :param \\*\\*kwargs:
        Keyword arguments that provide some options for the StyleGuide.
    :returns:
        An initialized StyleGuide
    """
    application = app.Application()
    application.plugins, application.options = parse_args([])
    options = application.options
    for key, value in kwargs.items():
        try:
            getattr(options, key)
            setattr(options, key, value)
        except AttributeError:
            LOG.error('Could not update option "%s"', key)
    application.make_formatter()
    application.make_guide()
    application.make_file_checker_manager([])
    return StyleGuide(application)