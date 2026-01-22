from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_format_child_option(self):
    br = branch.Branch.open('parent')
    conf = br.get_config_stack()
    conf.set('child_submit_format', '4')
    md = self.get_MD([])
    self.assertIs(merge_directive.MergeDirective2, md.__class__)
    conf.set('child_submit_format', '0.9')
    md = self.get_MD([])
    self.assertFormatIs(b'# Bazaar revision bundle v0.9', md)
    md = self.get_MD([], cmd=['bundle'])
    self.assertFormatIs(b'# Bazaar revision bundle v0.9', md)
    self.assertIs(merge_directive.MergeDirective, md.__class__)
    conf.set('child_submit_format', '0.999')
    self.run_bzr_error(["No such send format '0.999'"], 'send -f branch -o-')[0]