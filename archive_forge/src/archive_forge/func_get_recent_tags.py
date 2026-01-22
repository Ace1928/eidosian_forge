import datetime
import re
import sys
import time
from ..repo import Repo
def get_recent_tags(projdir=PROJDIR):
    """Get list of tags in order from newest to oldest and their datetimes.

    Args:
      projdir: path to ``.git``
    Returns:
      list of tags sorted by commit time from newest to oldest

    Each tag in the list contains the tag name, commit time, commit id, author
    and any tag meta. If a tag isn't annotated, then its tag meta is ``None``.
    Otherwise the tag meta is a tuple containing the tag time, tag id and tag
    name. Time is in UTC.
    """
    with Repo(projdir) as project:
        refs = project.get_refs()
        tags = {}
        for key, value in refs.items():
            key = key.decode('utf-8')
            obj = project.get_object(value)
            if 'tags' not in key:
                continue
            _, tag = key.rsplit('/', 1)
            try:
                commit = obj.object
            except AttributeError:
                commit = obj
                tag_meta = None
            else:
                tag_meta = (datetime.datetime(*time.gmtime(obj.tag_time)[:6]), obj.id.decode('utf-8'), obj.name.decode('utf-8'))
                commit = project.get_object(commit[1])
            tags[tag] = [datetime.datetime(*time.gmtime(commit.commit_time)[:6]), commit.id.decode('utf-8'), commit.author.decode('utf-8'), tag_meta]
    return sorted(tags.items(), key=lambda tag: tag[1][0], reverse=True)