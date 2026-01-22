import xml.dom.minidom
import subprocess
import os
from shutil import rmtree
import keyword
from ..base import (CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec,
import os\n\n\n"""
def grab_xml(module, launcher, mipav_hacks=False):
    command_list = launcher[:]
    command_list.extend([module, '--xml'])
    final_command = ' '.join(command_list)
    xmlReturnValue = subprocess.Popen(final_command, stdout=subprocess.PIPE, shell=True).communicate()[0]
    if mipav_hacks:
        new_xml = ''
        replace_closing_tag = False
        for line in xmlReturnValue.splitlines():
            if line.strip() == '<file collection: semi-colon delimited list>':
                new_xml += '<file-vector>\n'
                replace_closing_tag = True
            elif replace_closing_tag and line.strip() == '</file>':
                new_xml += '</file-vector>\n'
                replace_closing_tag = False
            else:
                new_xml += line + '\n'
        xmlReturnValue = new_xml
        if xmlReturnValue.strip().endswith('XML'):
            xmlReturnValue = xmlReturnValue.strip()[:-3]
        if xmlReturnValue.strip().startswith('Error: Unable to set default atlas'):
            xmlReturnValue = xmlReturnValue.strip()[len('Error: Unable to set default atlas'):]
    try:
        dom = xml.dom.minidom.parseString(xmlReturnValue.strip())
    except Exception as e:
        print(xmlReturnValue.strip())
        raise e
    return dom