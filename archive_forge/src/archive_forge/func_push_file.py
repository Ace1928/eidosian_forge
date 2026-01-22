import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
def push_file(self, xnat, file_name, out_key, uri_template_args):
    val_list = [unquote_id(val) for part in os.path.split(file_name)[0].split(os.sep) for val in part.split('_')[1:] if part.startswith('_') and len(part.split('_')) % 2]
    keymap = dict(list(zip(val_list[1::2], val_list[2::2])))
    _label = []
    for key, val in sorted(keymap.items()):
        if str(self.inputs.subject_id) not in val:
            _label.extend([key, val])
    uri_template_args['container_type'] = None
    for container in ['assessor_id', 'reconstruction_id']:
        if getattr(self.inputs, container):
            uri_template_args['container_type'] = container.split('_id')[0]
            uri_template_args['container_id'] = uri_template_args[container]
    if uri_template_args['container_type'] is None:
        uri_template_args['container_type'] = 'reconstruction'
        uri_template_args['container_id'] = unquote_id(uri_template_args['experiment_id'])
        if _label:
            uri_template_args['container_id'] += '_results_%s' % '_'.join(_label)
        else:
            uri_template_args['container_id'] += '_results'
    uri_template_args['resource_label'] = '%s_%s' % (uri_template_args['container_id'], out_key.split('.')[0])
    uri_template_args['file_name'] = os.path.split(os.path.abspath(unquote_id(file_name)))[1]
    uri_template = '/project/%(project_id)s/subject/%(subject_id)s/experiment/%(experiment_id)s/%(container_type)s/%(container_id)s/out/resource/%(resource_label)s/file/%(file_name)s'
    for key in list(uri_template_args.keys()):
        uri_template_args[key] = unquote_id(uri_template_args[key])
    remote_file = xnat.select(uri_template % uri_template_args)
    remote_file.insert(file_name, experiments='xnat:imageSessionData', use_label=True)
    if 'original_project' in uri_template_args:
        experiment_template = '/project/%(original_project)s/subject/%(subject_id)s/experiment/%(experiment_id)s'
        xnat.select(experiment_template % uri_template_args).share(uri_template_args['original_project'])