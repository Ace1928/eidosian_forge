import sys
import os
import io
import pathlib
import re
import argparse
import zipfile
import json
import pickle
import pprint
import urllib.parse
from typing import (
import torch.utils.show_pickle
def hierarchical_pickle(data):
    if isinstance(data, (bool, int, float, str, type(None))):
        return data
    if isinstance(data, list):
        return [hierarchical_pickle(d) for d in data]
    if isinstance(data, tuple):
        return {'__tuple_values__': hierarchical_pickle(list(data))}
    if isinstance(data, dict):
        return {'__is_dict__': True, 'keys': hierarchical_pickle(list(data.keys())), 'values': hierarchical_pickle(list(data.values()))}
    if isinstance(data, torch.utils.show_pickle.FakeObject):
        typename = f'{data.module}.{data.name}'
        if typename.startswith(('__torch__.', 'torch.jit.LoweredWrapper.', 'torch.jit.LoweredModule.')):
            assert data.args == ()
            return {'__module_type__': typename, 'state': hierarchical_pickle(data.state)}
        if typename == 'torch._utils._rebuild_tensor_v2':
            assert data.state is None
            if len(data.args) == 6:
                storage, offset, size, stride, requires_grad, hooks = data.args
            else:
                storage, offset, size, stride, requires_grad, hooks, metadata = data.args
            storage_info = get_storage_info(storage)
            return {'__tensor_v2__': [storage_info, offset, size, stride, requires_grad]}
        if typename == 'torch._utils._rebuild_qtensor':
            assert data.state is None
            storage, offset, size, stride, quantizer, requires_grad, hooks = data.args
            storage_info = get_storage_info(storage)
            assert isinstance(quantizer, tuple)
            assert isinstance(quantizer[0], torch.utils.show_pickle.FakeClass)
            assert quantizer[0].module == 'torch'
            if quantizer[0].name == 'per_tensor_affine':
                assert len(quantizer) == 3
                assert isinstance(quantizer[1], float)
                assert isinstance(quantizer[2], int)
                quantizer_extra = list(quantizer[1:3])
            else:
                quantizer_extra = []
            quantizer_json = [quantizer[0].name] + quantizer_extra
            return {'__qtensor__': [storage_info, offset, size, stride, quantizer_json, requires_grad]}
        if typename == 'torch.jit._pickle.restore_type_tag':
            assert data.state is None
            obj, typ = data.args
            assert isinstance(typ, str)
            return hierarchical_pickle(obj)
        if re.fullmatch('torch\\.jit\\._pickle\\.build_[a-z]+list', typename):
            assert data.state is None
            ls, = data.args
            assert isinstance(ls, list)
            return hierarchical_pickle(ls)
        if typename == 'torch.device':
            assert data.state is None
            name, = data.args
            assert isinstance(name, str)
            return name
        if typename == 'builtin.UnicodeDecodeError':
            assert data.state is None
            msg, = data.args
            assert isinstance(msg, str)
            return {'__module_type__': typename, 'state': hierarchical_pickle((msg,))}
        raise Exception(f"Can't prepare fake object of type for JS: {typename}")
    raise Exception(f"Can't prepare data of type for JS: {type(data)}")