import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def get_handler(self):
    from ..processors import generic_processor
    branch = self.make_branch('.', format=self.branch_format)
    handler = generic_processor.GenericProcessor(branch.controldir)
    return (handler, branch)