from __future__ import annotations
def groupby_tasks_group_hash(x, hash, grouper):
    return (hash(grouper(x)), x)