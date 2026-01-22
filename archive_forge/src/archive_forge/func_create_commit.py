import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def create_commit(data, marker=b'Default', blob=None):
    if not blob:
        blob = Blob.from_string(b'The blob content ' + marker)
    tree = Tree()
    tree.add(b'thefile_' + marker, 33188, blob.id)
    cmt = Commit()
    if data:
        assert isinstance(data[-1], Commit)
        cmt.parents = [data[-1].id]
    cmt.tree = tree.id
    author = b'John Doe ' + marker + b' <john@doe.net>'
    cmt.author = cmt.committer = author
    tz = parse_timezone(b'-0200')[0]
    cmt.commit_time = cmt.author_time = int(time())
    cmt.commit_timezone = cmt.author_timezone = tz
    cmt.encoding = b'UTF-8'
    cmt.message = b'The commit message ' + marker
    tag = Tag()
    tag.tagger = b'john@doe.net'
    tag.message = b'Annotated tag'
    tag.tag_timezone = parse_timezone(b'-0200')[0]
    tag.tag_time = cmt.author_time
    tag.object = (Commit, cmt.id)
    tag.name = b'v_' + marker + b'_0.1'
    return (blob, tree, tag, cmt)