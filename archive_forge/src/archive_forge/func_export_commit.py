import base64
import stat
from typing import Optional
import fastbencode as bencode
from .. import errors, foreign, trace, urlutils
from ..foreign import ForeignRevision, ForeignVcs, VcsMappingRegistry
from ..revision import NULL_REVISION, Revision
from .errors import NoPushSupport
from .hg import extract_hg_metadata, format_hg_metadata
from .roundtrip import (CommitSupplement, extract_bzr_metadata,
def export_commit(self, rev, tree_sha, parent_lookup, lossy, verifiers):
    """Turn a Bazaar revision in to a Git commit

        :param tree_sha: Tree sha for the commit
        :param parent_lookup: Function for looking up the GIT sha equiv of a
            bzr revision
        :param lossy: Whether to store roundtripping information.
        :param verifiers: Verifiers info
        :return dulwich.objects.Commit represent the revision:
        """
    from dulwich.objects import Commit, Tag
    commit = Commit()
    commit.tree = tree_sha
    if not lossy:
        metadata = CommitSupplement()
        metadata.verifiers = verifiers
    else:
        metadata = None
    parents = []
    for p in rev.parent_ids:
        try:
            git_p = parent_lookup(p)
        except KeyError:
            git_p = None
            if metadata is not None:
                metadata.explicit_parent_ids = rev.parent_ids
        if git_p is not None:
            if len(git_p) != 40:
                raise AssertionError('unexpected length for %r' % git_p)
            parents.append(git_p)
    commit.parents = parents
    try:
        encoding = rev.properties['git-explicit-encoding']
    except KeyError:
        encoding = rev.properties.get('git-implicit-encoding', 'utf-8')
    try:
        commit.encoding = rev.properties['git-explicit-encoding'].encode('ascii')
    except KeyError:
        pass
    commit.committer = fix_person_identifier(rev.committer.encode(encoding))
    first_author = rev.get_apparent_authors()[0]
    if ',' in first_author and first_author.count('>') > 1:
        first_author = first_author.split(',')[0]
    commit.author = fix_person_identifier(first_author.encode(encoding))
    long = getattr(__builtins__, 'long', int)
    commit.commit_time = long(rev.timestamp)
    if 'author-timestamp' in rev.properties:
        commit.author_time = long(rev.properties['author-timestamp'])
    else:
        commit.author_time = commit.commit_time
    commit._commit_timezone_neg_utc = 'commit-timezone-neg-utc' in rev.properties
    commit.commit_timezone = rev.timezone
    commit._author_timezone_neg_utc = 'author-timezone-neg-utc' in rev.properties
    if 'author-timezone' in rev.properties:
        commit.author_timezone = int(rev.properties['author-timezone'])
    else:
        commit.author_timezone = commit.commit_timezone
    if 'git-gpg-signature' in rev.properties:
        commit.gpgsig = rev.properties['git-gpg-signature'].encode('utf-8', 'surrogateescape')
    commit.message = self._encode_commit_message(rev, rev.message, encoding)
    if not isinstance(commit.message, bytes):
        raise TypeError(commit.message)
    if metadata is not None:
        try:
            mapping_registry.parse_revision_id(rev.revision_id)
        except errors.InvalidRevisionId:
            metadata.revision_id = rev.revision_id
        mapping_properties = {'author', 'author-timezone', 'author-timezone-neg-utc', 'commit-timezone-neg-utc', 'git-implicit-encoding', 'git-gpg-signature', 'git-explicit-encoding', 'author-timestamp', 'file-modes'}
        for k, v in rev.properties.items():
            if k not in mapping_properties:
                metadata.properties[k] = v
    if not lossy and metadata:
        if self.roundtripping:
            commit.message = inject_bzr_metadata(commit.message, metadata, encoding)
        else:
            raise NoPushSupport(None, None, self, revision_id=rev.revision_id)
    if not isinstance(commit.message, bytes):
        raise TypeError(commit.message)
    i = 0
    propname = 'git-mergetag-0'
    while propname in rev.properties:
        commit.mergetag.append(Tag.from_string(rev.properties[propname].encode('utf-8', 'surrogateescape')))
        i += 1
        propname = 'git-mergetag-%d' % i
    try:
        extra = commit._extra
    except AttributeError:
        extra = commit.extra
    if 'git-extra' in rev.properties:
        for l in rev.properties['git-extra'].splitlines():
            k, v = l.split(' ', 1)
            extra.append((k.encode('utf-8', 'surrogateescape'), v.encode('utf-8', 'surrogateescape')))
    return commit