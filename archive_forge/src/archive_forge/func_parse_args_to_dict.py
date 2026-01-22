import abc
import argparse
import functools
import logging
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def parse_args_to_dict(values_specs):
    """It is used to analyze the extra command options to command.

    Besides known options and arguments, our commands also support user to
    put more options to the end of command line. For example,
    list_nets -- --tag x y --key1 value1, where '-- --tag x y --key1 value1'
    is extra options to our list_nets. This feature can support V2.0 API's
    fields selection and filters. For example, to list networks which has name
    'test4', we can have list_nets -- --name=test4.

    value spec is: --key type=int|bool|... value. Type is one of Python
    built-in types. By default, type is string. The key without value is
    a bool option. Key with two values will be a list option.
    """
    values_specs_copy = values_specs[:]
    if values_specs_copy and values_specs_copy[0] == '--':
        del values_specs_copy[0]
    _options = {}
    current_arg = None
    _values_specs = []
    _value_number = 0
    _list_flag = False
    _clear_flag = False
    current_item = None
    current_type_str = None
    allowed_type_dict = {'bool': utils.str2bool, 'dict': utils.str2dict, 'int': int, 'str': str}
    for _item in values_specs_copy:
        if _item.startswith('--'):
            _process_previous_argument(current_arg, _value_number, current_type_str, _list_flag, _values_specs, _clear_flag, values_specs)
            current_item = _item
            _list_flag = False
            _clear_flag = False
            current_type_str = None
            if '=' in _item:
                _value_number = 1
                _item = _item.split('=')[0]
            else:
                _value_number = 0
            if _item in _options:
                raise exceptions.CommandError(_('Duplicated options %s') % ' '.join(values_specs))
            else:
                _options.update({_item: {}})
            current_arg = _options[_item]
            _item = current_item
        elif _item.startswith('type='):
            if current_arg is None:
                raise exceptions.CommandError(_('Invalid values_specs %s') % ' '.join(values_specs))
            if 'type' not in current_arg:
                current_type_str = _item.split('=', 2)[1]
                if current_type_str in allowed_type_dict:
                    current_arg['type'] = allowed_type_dict[current_type_str]
                    continue
                else:
                    raise exceptions.CommandError(_('Invalid value_specs {valspec}: type {curtypestr} is not supported').format(valspec=' '.join(values_specs), curtypestr=current_type_str))
        elif _item == 'list=true':
            _list_flag = True
            continue
        elif _item == 'action=clear':
            _clear_flag = True
            continue
        if not _item.startswith('--'):
            if not current_item or '=' in current_item or (_item.startswith('-') and (not is_number(_item))):
                raise exceptions.CommandError(_('Invalid values_specs %s') % ' '.join(values_specs))
            _value_number += 1
        if _item.startswith('---'):
            raise exceptions.CommandError(_('Invalid values_specs %s') % ' '.join(values_specs))
        _values_specs.append(_item)
    _process_previous_argument(current_arg, _value_number, current_type_str, _list_flag, _values_specs, _clear_flag, values_specs)
    _parser = argparse.ArgumentParser(add_help=False)
    for opt, optspec in _options.items():
        _parser.add_argument(opt, **optspec)
    _args = _parser.parse_args(_values_specs)
    result_dict = {}
    for opt in _options.keys():
        _opt = opt.split('--', 2)[1]
        _opt = _opt.replace('-', '_')
        _value = getattr(_args, _opt)
        result_dict.update({_opt: _value})
    return result_dict