import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def write_workflow_prov(graph, filename=None, format='all'):
    """Write W3C PROV Model JSON file"""
    if not filename:
        filename = os.path.join(os.getcwd(), 'workflow_provenance')
    ps = ProvStore()
    processes = []
    nodes = graph.nodes()
    for node in nodes:
        result = node.result
        classname = node.interface.__class__.__name__
        _, hashval, _, _ = node.hash_exists()
        attrs = {pm.PROV['type']: nipype_ns[classname], pm.PROV['label']: '_'.join((classname, node.name)), nipype_ns['hashval']: hashval}
        process = ps.g.activity(get_id(), None, None, attrs)
        if isinstance(result.runtime, list):
            process.add_attributes({pm.PROV['type']: nipype_ns['MapNode']})
            for idx, runtime in enumerate(result.runtime):
                subresult = InterfaceResult(result.interface[idx], runtime, outputs={})
                if result.inputs:
                    if idx < len(result.inputs):
                        subresult.inputs = result.inputs[idx]
                if result.outputs:
                    for key, _ in list(result.outputs.items()):
                        values = getattr(result.outputs, key)
                        if isdefined(values) and idx < len(values):
                            subresult.outputs[key] = values[idx]
                sub_doc = ProvStore().add_results(subresult)
                sub_bundle = pm.ProvBundle(sub_doc.get_records(), identifier=get_id())
                ps.g.add_bundle(sub_bundle)
                bundle_entity = ps.g.entity(sub_bundle.identifier, other_attributes={'prov:type': pm.PROV_BUNDLE})
                ps.g.wasGeneratedBy(bundle_entity, process)
        else:
            process.add_attributes({pm.PROV['type']: nipype_ns['Node']})
            if result.provenance:
                prov_doc = result.provenance
            else:
                prov_doc = ProvStore().add_results(result)
            result_bundle = pm.ProvBundle(prov_doc.get_records(), identifier=get_id())
            ps.g.add_bundle(result_bundle)
            bundle_entity = ps.g.entity(result_bundle.identifier, other_attributes={'prov:type': pm.PROV_BUNDLE})
            ps.g.wasGeneratedBy(bundle_entity, process)
        processes.append(process)
    for idx, edgeinfo in enumerate(graph.in_edges()):
        ps.g.wasStartedBy(processes[list(nodes).index(edgeinfo[1])], starter=processes[list(nodes).index(edgeinfo[0])])
    ps.write_provenance(filename, format=format)
    return ps.g