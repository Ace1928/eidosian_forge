import time
import platform
import socket
from lxml import etree
from lxml.etree import Element, QName
from .uriutil import uri_parent
from .jsonutil import JsonTable
from . import httputil
def process_step_xml(**kwargs):
    step_node = Element(QName(_nsmap['prov'], 'processStep'), nsmap=_nsmap)
    program_node = Element(QName(_nsmap['prov'], 'program'), nsmap=_nsmap)
    program_node.text = kwargs['program']
    if 'program_version' in kwargs.keys():
        program_node.set('version', kwargs['program_version'])
    if 'program_arguments' in kwargs.keys():
        program_node.set('arguments', kwargs['program_arguments'])
    step_node.append(program_node)
    timestamp_node = Element(QName(_nsmap['prov'], 'timestamp'), nsmap=_nsmap)
    timestamp_node.text = kwargs['timestamp']
    step_node.append(timestamp_node)
    if 'cvs' in kwargs.keys():
        cvs_node = Element(QName(_nsmap['prov'], 'cvs'), nsmap=_nsmap)
        cvs_node.text = kwargs['cvs']
        step_node.append(cvs_node)
    user_node = Element(QName(_nsmap['prov'], 'user'), nsmap=_nsmap)
    user_node.text = kwargs['user']
    step_node.append(user_node)
    machine_node = Element(QName(_nsmap['prov'], 'machine'), nsmap=_nsmap)
    machine_node.text = kwargs['machine']
    step_node.append(machine_node)
    platform_node = Element(QName(_nsmap['prov'], 'platform'), nsmap=_nsmap)
    platform_node.text = kwargs['platform']
    if 'platform_version' in kwargs.keys():
        platform_node.set('version', kwargs['platform_version'])
    step_node.append(platform_node)
    if 'compiler' in kwargs.keys():
        compiler_node = Element(QName(_nsmap['prov'], 'compiler'), nsmap=_nsmap)
        compiler_node.text = kwargs['compiler']
        if 'compiler_version' in kwargs.keys():
            compiler_node.set('version', kwargs['compiler_version'])
        step_node.append(compiler_node)
    if 'library' in kwargs.keys():
        library_node = Element(QName(_nsmap['prov'], 'library'), nsmap=_nsmap)
        library_node.text = kwargs['library']
        if 'library_version' in kwargs.keys():
            library_node.set('version', kwargs['library_version'])
        step_node.append(library_node)
    return step_node