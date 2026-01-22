import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import
def formXML(self, stub):
    solver = self.getSolverName()
    zipped_nl_file = io.BytesIO()
    if os.path.exists(stub) and stub[-3:] == '.nl':
        stub = stub[:-3]
    nlfile = open(stub + '.nl', 'rb')
    zipper = gzip.GzipFile(mode='wb', fileobj=zipped_nl_file)
    zipper.write(nlfile.read())
    zipper.close()
    nlfile.close()
    ampl_files = {}
    for key in ['adj', 'col', 'env', 'fix', 'spc', 'row', 'slc', 'unv']:
        if os.access(stub + '.' + key, os.R_OK):
            f = open(stub + '.' + key, 'r')
            val = ''
            buf = f.read()
            while buf:
                val += buf
                buf = f.read()
            f.close()
            ampl_files[key] = val
    priority = ''
    m = re.search('priority[\\s=]+(\\S+)', self.options)
    if m:
        priority = '<priority>%s</priority>\n' % m.groups()[0]
    solver_options = 'kestrel_options:solver=%s\n' % solver.lower()
    solver_options_key = '%s_options' % solver
    solver_options_value = ''
    if solver_options_key in os.environ:
        solver_options_value = os.getenv(solver_options_key)
    elif solver_options_key.lower() in os.environ:
        solver_options_value = os.getenv(solver_options_key.lower())
    elif solver_options_key.upper() in os.environ:
        solver_options_value = os.getenv(solver_options_key.upper())
    if not solver_options_value == '':
        solver_options += '%s_options:%s\n' % (solver.lower(), solver_options_value)
    nl_string = base64.encodebytes(zipped_nl_file.getvalue()).decode('utf-8')
    xml = '\n              <document>\n              <category>kestrel</category>\n              <email>%s</email>\n              <solver>%s</solver>\n              <inputType>AMPL</inputType>\n              %s\n              <solver_options>%s</solver_options>\n              <nlfile><base64>%s</base64></nlfile>\n' % (self.getEmailAddress(), solver, priority, solver_options, nl_string)
    for key in ampl_files:
        xml += '<%s><![CDATA[%s]]></%s>\n' % (key, ampl_files[key], key)
    for option in ['kestrel_auxfiles', 'mip_priorities', 'objective_precision']:
        if option in os.environ:
            xml += '<%s><![CDATA[%s]]></%s>\n' % (option, os.getenv(option), option)
    xml += '</document>'
    return xml