import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
def test_registered_methods(self):
    """Test that known methods are registered to the correct object."""
    self.assertHandlerEqual(b'Branch.break_lock', smart_branch.SmartServerBranchBreakLock)
    self.assertHandlerEqual(b'Branch.get_config_file', smart_branch.SmartServerBranchGetConfigFile)
    self.assertHandlerEqual(b'Branch.put_config_file', smart_branch.SmartServerBranchPutConfigFile)
    self.assertHandlerEqual(b'Branch.get_parent', smart_branch.SmartServerBranchGetParent)
    self.assertHandlerEqual(b'Branch.get_physical_lock_status', smart_branch.SmartServerBranchRequestGetPhysicalLockStatus)
    self.assertHandlerEqual(b'Branch.get_tags_bytes', smart_branch.SmartServerBranchGetTagsBytes)
    self.assertHandlerEqual(b'Branch.lock_write', smart_branch.SmartServerBranchRequestLockWrite)
    self.assertHandlerEqual(b'Branch.last_revision_info', smart_branch.SmartServerBranchRequestLastRevisionInfo)
    self.assertHandlerEqual(b'Branch.revision_history', smart_branch.SmartServerRequestRevisionHistory)
    self.assertHandlerEqual(b'Branch.revision_id_to_revno', smart_branch.SmartServerBranchRequestRevisionIdToRevno)
    self.assertHandlerEqual(b'Branch.set_config_option', smart_branch.SmartServerBranchRequestSetConfigOption)
    self.assertHandlerEqual(b'Branch.set_last_revision', smart_branch.SmartServerBranchRequestSetLastRevision)
    self.assertHandlerEqual(b'Branch.set_last_revision_info', smart_branch.SmartServerBranchRequestSetLastRevisionInfo)
    self.assertHandlerEqual(b'Branch.set_last_revision_ex', smart_branch.SmartServerBranchRequestSetLastRevisionEx)
    self.assertHandlerEqual(b'Branch.set_parent_location', smart_branch.SmartServerBranchRequestSetParentLocation)
    self.assertHandlerEqual(b'Branch.unlock', smart_branch.SmartServerBranchRequestUnlock)
    self.assertHandlerEqual(b'BzrDir.destroy_branch', smart_dir.SmartServerBzrDirRequestDestroyBranch)
    self.assertHandlerEqual(b'BzrDir.find_repository', smart_dir.SmartServerRequestFindRepositoryV1)
    self.assertHandlerEqual(b'BzrDir.find_repositoryV2', smart_dir.SmartServerRequestFindRepositoryV2)
    self.assertHandlerEqual(b'BzrDirFormat.initialize', smart_dir.SmartServerRequestInitializeBzrDir)
    self.assertHandlerEqual(b'BzrDirFormat.initialize_ex_1.16', smart_dir.SmartServerRequestBzrDirInitializeEx)
    self.assertHandlerEqual(b'BzrDir.checkout_metadir', smart_dir.SmartServerBzrDirRequestCheckoutMetaDir)
    self.assertHandlerEqual(b'BzrDir.cloning_metadir', smart_dir.SmartServerBzrDirRequestCloningMetaDir)
    self.assertHandlerEqual(b'BzrDir.get_branches', smart_dir.SmartServerBzrDirRequestGetBranches)
    self.assertHandlerEqual(b'BzrDir.get_config_file', smart_dir.SmartServerBzrDirRequestConfigFile)
    self.assertHandlerEqual(b'BzrDir.open_branch', smart_dir.SmartServerRequestOpenBranch)
    self.assertHandlerEqual(b'BzrDir.open_branchV2', smart_dir.SmartServerRequestOpenBranchV2)
    self.assertHandlerEqual(b'BzrDir.open_branchV3', smart_dir.SmartServerRequestOpenBranchV3)
    self.assertHandlerEqual(b'PackRepository.autopack', smart_packrepo.SmartServerPackRepositoryAutopack)
    self.assertHandlerEqual(b'Repository.add_signature_text', smart_repo.SmartServerRepositoryAddSignatureText)
    self.assertHandlerEqual(b'Repository.all_revision_ids', smart_repo.SmartServerRepositoryAllRevisionIds)
    self.assertHandlerEqual(b'Repository.break_lock', smart_repo.SmartServerRepositoryBreakLock)
    self.assertHandlerEqual(b'Repository.gather_stats', smart_repo.SmartServerRepositoryGatherStats)
    self.assertHandlerEqual(b'Repository.get_parent_map', smart_repo.SmartServerRepositoryGetParentMap)
    self.assertHandlerEqual(b'Repository.get_physical_lock_status', smart_repo.SmartServerRepositoryGetPhysicalLockStatus)
    self.assertHandlerEqual(b'Repository.get_rev_id_for_revno', smart_repo.SmartServerRepositoryGetRevIdForRevno)
    self.assertHandlerEqual(b'Repository.get_revision_graph', smart_repo.SmartServerRepositoryGetRevisionGraph)
    self.assertHandlerEqual(b'Repository.get_revision_signature_text', smart_repo.SmartServerRepositoryGetRevisionSignatureText)
    self.assertHandlerEqual(b'Repository.get_stream', smart_repo.SmartServerRepositoryGetStream)
    self.assertHandlerEqual(b'Repository.get_stream_1.19', smart_repo.SmartServerRepositoryGetStream_1_19)
    self.assertHandlerEqual(b'Repository.iter_revisions', smart_repo.SmartServerRepositoryIterRevisions)
    self.assertHandlerEqual(b'Repository.has_revision', smart_repo.SmartServerRequestHasRevision)
    self.assertHandlerEqual(b'Repository.insert_stream', smart_repo.SmartServerRepositoryInsertStream)
    self.assertHandlerEqual(b'Repository.insert_stream_locked', smart_repo.SmartServerRepositoryInsertStreamLocked)
    self.assertHandlerEqual(b'Repository.is_shared', smart_repo.SmartServerRepositoryIsShared)
    self.assertHandlerEqual(b'Repository.iter_files_bytes', smart_repo.SmartServerRepositoryIterFilesBytes)
    self.assertHandlerEqual(b'Repository.lock_write', smart_repo.SmartServerRepositoryLockWrite)
    self.assertHandlerEqual(b'Repository.make_working_trees', smart_repo.SmartServerRepositoryMakeWorkingTrees)
    self.assertHandlerEqual(b'Repository.pack', smart_repo.SmartServerRepositoryPack)
    self.assertHandlerEqual(b'Repository.reconcile', smart_repo.SmartServerRepositoryReconcile)
    self.assertHandlerEqual(b'Repository.tarball', smart_repo.SmartServerRepositoryTarball)
    self.assertHandlerEqual(b'Repository.unlock', smart_repo.SmartServerRepositoryUnlock)
    self.assertHandlerEqual(b'Repository.start_write_group', smart_repo.SmartServerRepositoryStartWriteGroup)
    self.assertHandlerEqual(b'Repository.check_write_group', smart_repo.SmartServerRepositoryCheckWriteGroup)
    self.assertHandlerEqual(b'Repository.commit_write_group', smart_repo.SmartServerRepositoryCommitWriteGroup)
    self.assertHandlerEqual(b'Repository.abort_write_group', smart_repo.SmartServerRepositoryAbortWriteGroup)
    self.assertHandlerEqual(b'VersionedFileRepository.get_serializer_format', smart_repo.SmartServerRepositoryGetSerializerFormat)
    self.assertHandlerEqual(b'VersionedFileRepository.get_inventories', smart_repo.SmartServerRepositoryGetInventories)
    self.assertHandlerEqual(b'Transport.is_readonly', smart_req.SmartServerIsReadonly)