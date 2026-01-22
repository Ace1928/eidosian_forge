from __future__ import (absolute_import, division, print_function)
from functools import wraps
from os import environ
from os import path
from datetime import datetime
@api_wrapper
def get_export(module, system):
    """Return export if found or None if not found"""
    try:
        try:
            export_name = module.params['export']
        except KeyError:
            export_name = module.params['name']
        export = system.exports.get(export_path=export_name)
    except ObjectNotFound:
        return None
    return export