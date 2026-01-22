import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_describe_checkout_format(self):
    for key in controldir.format_registry.keys():
        if key in controldir.format_registry.aliases():
            continue
        if key == 'weave':
            continue
        if key in ('git', 'git-bare'):
            continue
        if controldir.format_registry.get_info(key).experimental:
            continue
        if controldir.format_registry.get_info(key).hidden:
            continue
        expected = None
        if key in ('pack-0.92',):
            expected = 'pack-0.92'
        elif key in ('knit', 'metaweave'):
            if 'metaweave' in controldir.format_registry:
                expected = 'knit or metaweave'
            else:
                expected = 'knit'
        elif key in ('1.14', '1.14-rich-root'):
            expected = '1.14 or 1.14-rich-root'
        self.assertCheckoutDescription(key, expected)