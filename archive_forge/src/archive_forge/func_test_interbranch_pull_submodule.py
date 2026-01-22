import os
import dulwich
from dulwich.objects import Commit, Tag
from dulwich.repo import Repo as GitRepo
from ... import errors, revision, urlutils
from ...branch import Branch, InterBranch, UnstackableBranchFormat
from ...controldir import ControlDir
from ...repository import Repository
from .. import branch, tests
from ..dir import LocalGitControlDirFormat
from ..mapping import default_mapping
def test_interbranch_pull_submodule(self):
    path = 'd'
    os.mkdir(path)
    os.chdir(path)
    GitRepo.init('.')
    bb = tests.GitBranchBuilder()
    bb.set_file('foobar', b'foo\nbar\n', False)
    mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
    bb.set_submodule('core', b'102ee7206ebc4227bec8ac02450972e6738f4a33')
    bb.set_file('.gitmodules', b'[submodule "core"]\n  path = core\n  url = https://github.com/phhusson/QuasselC.git\n', False)
    mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
    marks = bb.finish()
    os.chdir('..')
    gitsha1 = marks[mark1]
    gitsha2 = marks[mark2]
    oldrepo = Repository.open(path)
    revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
    newbranch = self.make_branch('g')
    inter_branch = InterBranch.get(Branch.open(path), newbranch)
    inter_branch.pull()
    self.assertEqual(revid2, newbranch.last_revision())
    self.assertEqual(('https://github.com/phhusson/QuasselC.git', 'core'), newbranch.get_reference_info(newbranch.basis_tree().path2id('core')))