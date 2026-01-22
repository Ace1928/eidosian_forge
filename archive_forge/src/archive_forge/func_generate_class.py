import xml.dom.minidom
import subprocess
import os
from shutil import rmtree
import keyword
from ..base import (CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec,
import os\n\n\n"""
def generate_class(module, launcher, strip_module_name_prefix=True, redirect_x=False, mipav_hacks=False):
    dom = grab_xml(module, launcher, mipav_hacks=mipav_hacks)
    if strip_module_name_prefix:
        module_name = module.split('.')[-1]
    else:
        module_name = module
    inputTraits = []
    outputTraits = []
    outputs_filenames = {}
    class_string = '"""'
    for desc_str in ['title', 'category', 'description', 'version', 'documentation-url', 'license', 'contributor', 'acknowledgements']:
        el = dom.getElementsByTagName(desc_str)
        if el and el[0].firstChild and el[0].firstChild.nodeValue.strip():
            class_string += desc_str + ': ' + el[0].firstChild.nodeValue.strip() + '\n\n'
        if desc_str == 'category':
            category = el[0].firstChild.nodeValue.strip()
    class_string += '"""'
    for paramGroup in dom.getElementsByTagName('parameters'):
        indices = paramGroup.getElementsByTagName('index')
        max_index = 0
        for index in indices:
            if int(index.firstChild.nodeValue) > max_index:
                max_index = int(index.firstChild.nodeValue)
        for param in paramGroup.childNodes:
            if param.nodeName in ['label', 'description', '#text', '#comment']:
                continue
            traitsParams = {}
            longFlagNode = param.getElementsByTagName('longflag')
            if longFlagNode:
                longFlagName = longFlagNode[0].firstChild.nodeValue
                longFlagName = longFlagName.lstrip(' -').rstrip(' ')
                name = longFlagName
                name = force_to_valid_python_variable_name(name)
                traitsParams['argstr'] = '--' + longFlagName + ' '
            else:
                name = param.getElementsByTagName('name')[0].firstChild.nodeValue
                name = force_to_valid_python_variable_name(name)
                if param.getElementsByTagName('index'):
                    traitsParams['argstr'] = ''
                else:
                    traitsParams['argstr'] = '--' + name + ' '
            if param.getElementsByTagName('description') and param.getElementsByTagName('description')[0].firstChild:
                traitsParams['desc'] = param.getElementsByTagName('description')[0].firstChild.nodeValue.replace('"', '\\"').replace('\n', ', ')
            argsDict = {'directory': '%s', 'file': '%s', 'integer': '%d', 'double': '%f', 'float': '%f', 'image': '%s', 'transform': '%s', 'boolean': '', 'string-enumeration': '%s', 'string': '%s', 'integer-enumeration': '%s', 'table': '%s', 'point': '%s', 'region': '%s', 'geometry': '%s'}
            if param.nodeName.endswith('-vector'):
                traitsParams['argstr'] += '%s'
            else:
                traitsParams['argstr'] += argsDict[param.nodeName]
            index = param.getElementsByTagName('index')
            if index:
                traitsParams['position'] = int(index[0].firstChild.nodeValue) - (max_index + 1)
            desc = param.getElementsByTagName('description')
            if index:
                traitsParams['desc'] = desc[0].firstChild.nodeValue
            typesDict = {'integer': 'traits.Int', 'double': 'traits.Float', 'float': 'traits.Float', 'image': 'File', 'transform': 'File', 'boolean': 'traits.Bool', 'string': 'traits.Str', 'file': 'File', 'geometry': 'File', 'directory': 'Directory', 'table': 'File', 'point': 'traits.List', 'region': 'traits.List'}
            if param.nodeName.endswith('-enumeration'):
                type = 'traits.Enum'
                values = ['"%s"' % str(el.firstChild.nodeValue).replace('"', '') for el in param.getElementsByTagName('element')]
            elif param.nodeName.endswith('-vector'):
                type = 'InputMultiPath'
                if param.nodeName in ['file', 'directory', 'image', 'geometry', 'transform', 'table']:
                    values = ['%s(exists=True)' % typesDict[param.nodeName.replace('-vector', '')]]
                else:
                    values = [typesDict[param.nodeName.replace('-vector', '')]]
                if mipav_hacks is True:
                    traitsParams['sep'] = ';'
                else:
                    traitsParams['sep'] = ','
            elif param.getAttribute('multiple') == 'true':
                type = 'InputMultiPath'
                if param.nodeName in ['file', 'directory', 'image', 'geometry', 'transform', 'table']:
                    values = ['%s(exists=True)' % typesDict[param.nodeName]]
                elif param.nodeName in ['point', 'region']:
                    values = ['%s(traits.Float(), minlen=3, maxlen=3)' % typesDict[param.nodeName]]
                else:
                    values = [typesDict[param.nodeName]]
                traitsParams['argstr'] += '...'
            else:
                values = []
                type = typesDict[param.nodeName]
            if param.nodeName in ['file', 'directory', 'image', 'geometry', 'transform', 'table']:
                if not param.getElementsByTagName('channel'):
                    raise RuntimeError("Insufficient XML specification: each element of type 'file', 'directory', 'image', 'geometry', 'transform',  or 'table' requires 'channel' field.\n{0}".format(traitsParams))
                elif param.getElementsByTagName('channel')[0].firstChild.nodeValue == 'output':
                    traitsParams['hash_files'] = False
                    inputTraits.append('%s = traits.Either(traits.Bool, %s(%s), %s)' % (name, type, parse_values(values).replace('exists=True', ''), parse_params(traitsParams)))
                    traitsParams['exists'] = True
                    traitsParams.pop('argstr')
                    traitsParams.pop('hash_files')
                    outputTraits.append('%s = %s(%s%s)' % (name, type.replace('Input', 'Output'), parse_values(values), parse_params(traitsParams)))
                    outputs_filenames[name] = gen_filename_from_param(param, name)
                elif param.getElementsByTagName('channel')[0].firstChild.nodeValue == 'input':
                    if param.nodeName in ['file', 'directory', 'image', 'geometry', 'transform', 'table'] and type not in ['InputMultiPath', 'traits.List']:
                        traitsParams['exists'] = True
                    inputTraits.append('%s = %s(%s%s)' % (name, type, parse_values(values), parse_params(traitsParams)))
                else:
                    raise RuntimeError("Insufficient XML specification: each element of type 'file', 'directory', 'image', 'geometry', 'transform',  or 'table' requires 'channel' field to be in ['input','output'].\n{0}".format(traitsParams))
            else:
                inputTraits.append('%s = %s(%s%s)' % (name, type, parse_values(values), parse_params(traitsParams)))
    if mipav_hacks:
        blacklisted_inputs = ['maxMemoryUsage']
        inputTraits = [trait for trait in inputTraits if trait.split()[0] not in blacklisted_inputs]
        compulsory_inputs = ['xDefaultMem = traits.Int(desc="Set default maximum heap size", argstr="-xDefaultMem %d")', 'xMaxProcess = traits.Int(1, desc="Set default maximum number of processes.", argstr="-xMaxProcess %d", usedefault=True)']
        inputTraits += compulsory_inputs
    input_spec_code = 'class ' + module_name + 'InputSpec(CommandLineInputSpec):\n'
    for trait in inputTraits:
        input_spec_code += '    ' + trait + '\n'
    output_spec_code = 'class ' + module_name + 'OutputSpec(TraitedSpec):\n'
    if not outputTraits:
        output_spec_code += '    pass\n'
    else:
        for trait in outputTraits:
            output_spec_code += '    ' + trait + '\n'
    output_filenames_code = '_outputs_filenames = {'
    output_filenames_code += ','.join(["'%s':'%s'" % (key, value) for key, value in outputs_filenames.items()])
    output_filenames_code += '}'
    input_spec_code += '\n\n'
    output_spec_code += '\n\n'
    template = 'class %module_name%(SEMLikeCommandLine):\n    %class_str%\n\n    input_spec = %module_name%InputSpec\n    output_spec = %module_name%OutputSpec\n    _cmd = "%launcher% %name% "\n    %output_filenames_code%\n'
    template += '    _redirect_x = {0}\n'.format(str(redirect_x))
    main_class = template.replace('%class_str%', class_string).replace('%module_name%', module_name).replace('%name%', module).replace('%output_filenames_code%', output_filenames_code).replace('%launcher%', ' '.join(launcher))
    return (category, input_spec_code + output_spec_code + main_class, module_name)