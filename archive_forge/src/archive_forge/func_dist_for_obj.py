import argparse
import inspect
import traceback
import autopage.argparse
from . import command
def dist_for_obj(obj):
    name = inspect.getmodule(obj).__name__.partition('.')[0]
    return dists_by_module.get(name)