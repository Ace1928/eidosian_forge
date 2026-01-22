import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import graph_flow as gf
from taskflow import task
import example_utils as eu  # noqa
def make_flow_and_store(source_files, executable_only=False):
    flow = gf.TargetedFlow('build-flow')
    object_targets = []
    store = {}
    for source in source_files:
        source_stored = '%s-source' % source
        object_stored = '%s-object' % source
        store[source_stored] = source
        object_targets.append(object_stored)
        flow.add(CompileTask(name='compile-%s' % source, rebind={'source_filename': source_stored}, provides=object_stored))
    flow.add(BuildDocsTask(requires=list(store.keys())))
    object_targets.append('docs')
    link_task = LinkTask('build/executable', requires=object_targets)
    flow.add(link_task)
    if executable_only:
        flow.set_target(link_task)
    return (flow, store)