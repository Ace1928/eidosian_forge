import inspect
import json
import re
from typing import Callable, Optional
from jsonschema.protocols import Validator
from pydantic import create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012
def to_regex(resolver: Resolver, instance: dict, whitespace_pattern: Optional[str]=None):
    """Translate a JSON Schema instance into a regex that validates the schema.

    Note
    ----
    Many features of JSON schema are missing:
    - Handle `additionalProperties` keyword
    - Handle types defined as a list
    - Handle constraints on numbers
    - Handle special patterns: `date`, `uri`, etc.

    This does not support recursive definitions.

    Parameters
    ----------
    resolver
        An object that resolves references to other instances within a schema
    instance
        The instance to translate
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[
 ]?"`
    """
    if whitespace_pattern is None:
        whitespace_pattern = WHITESPACE
    if 'properties' in instance:
        regex = ''
        regex += '\\{'
        properties = instance['properties']
        required_properties = instance.get('required', [])
        is_required = [item in required_properties for item in properties]
        if any(is_required):
            last_required_pos = max([i for i, value in enumerate(is_required) if value])
            for i, (name, value) in enumerate(properties.items()):
                subregex = f'{whitespace_pattern}"{re.escape(name)}"{whitespace_pattern}:{whitespace_pattern}'
                subregex += to_regex(resolver, value, whitespace_pattern)
                if i < last_required_pos:
                    subregex = f'{subregex}{whitespace_pattern},'
                elif i > last_required_pos:
                    subregex = f'{whitespace_pattern},{subregex}'
                regex += subregex if is_required[i] else f'({subregex})?'
        else:
            property_subregexes = []
            for i, (name, value) in enumerate(properties.items()):
                subregex = f'{whitespace_pattern}"{name}"{whitespace_pattern}:{whitespace_pattern}'
                subregex += to_regex(resolver, value, whitespace_pattern)
                property_subregexes.append(subregex)
            possible_patterns = []
            for i in range(len(property_subregexes)):
                pattern = ''
                for subregex in property_subregexes[:i]:
                    pattern += f'({subregex}{whitespace_pattern},)?'
                pattern += property_subregexes[i]
                for subregex in property_subregexes[i + 1:]:
                    pattern += f'({whitespace_pattern},{subregex})?'
                possible_patterns.append(pattern)
            regex += f'({'|'.join(possible_patterns)})?'
        regex += f'{whitespace_pattern}' + '\\}'
        return regex
    elif 'allOf' in instance:
        subregexes = [to_regex(resolver, t, whitespace_pattern) for t in instance['allOf']]
        subregexes_str = [f'{subregex}' for subregex in subregexes]
        return f'({''.join(subregexes_str)})'
    elif 'anyOf' in instance:
        subregexes = [to_regex(resolver, t, whitespace_pattern) for t in instance['anyOf']]
        return f'({'|'.join(subregexes)})'
    elif 'oneOf' in instance:
        subregexes = [to_regex(resolver, t, whitespace_pattern) for t in instance['oneOf']]
        xor_patterns = []
        for subregex in subregexes:
            other_subregexes = filter(lambda r: r != subregex, subregexes)
            other_subregexes_str = '|'.join([f'{s}' for s in other_subregexes])
            negative_lookahead = f'(?!.*({other_subregexes_str}))'
            xor_patterns.append(f'({subregex}){negative_lookahead}')
        return f'({'|'.join(xor_patterns)})'
    elif 'enum' in instance:
        choices = []
        for choice in instance['enum']:
            if type(choice) in [int, float, bool, None]:
                choices.append(re.escape(str(choice)))
            elif type(choice) == str:
                choices.append(f'"{re.escape(choice)}"')
        return f'({'|'.join(choices)})'
    elif 'const' in instance:
        const = instance['const']
        if type(const) in [int, float, bool, None]:
            const = re.escape(str(const))
        elif type(const) == str:
            const = f'"{re.escape(const)}"'
        return const
    elif '$ref' in instance:
        path = f'{instance['$ref']}'
        instance = resolver.lookup(path).contents
        return to_regex(resolver, instance, whitespace_pattern)
    elif 'type' in instance:
        instance_type = instance['type']
        if instance_type == 'string':
            if 'maxLength' in instance or 'minLength' in instance:
                max_items = instance.get('maxLength', '')
                min_items = instance.get('minLength', '')
                try:
                    if int(max_items) < int(min_items):
                        raise ValueError('maxLength must be greater than or equal to minLength')
                except ValueError:
                    pass
                return f'"{STRING_INNER}{{{min_items},{max_items}}}"'
            elif 'pattern' in instance:
                pattern = instance['pattern']
                if pattern[0] == '^' and pattern[-1] == '$':
                    return f'(^"{pattern[1:-1]}"$)'
                else:
                    return f'("{pattern}")'
            elif 'format' in instance:
                format = instance['format']
                if format == 'date-time':
                    return format_to_regex['date-time']
                elif format == 'uuid':
                    return format_to_regex['uuid']
                elif format == 'date':
                    return format_to_regex['date']
                elif format == 'time':
                    return format_to_regex['time']
                else:
                    raise NotImplementedError(f'Format {format} is not supported by Outlines')
            else:
                return type_to_regex['string']
        elif instance_type == 'number':
            return type_to_regex['number']
        elif instance_type == 'integer':
            return type_to_regex['integer']
        elif instance_type == 'array':
            num_repeats = _get_num_items_pattern(instance.get('minItems'), instance.get('maxItems'), whitespace_pattern)
            if num_repeats is None:
                return f'\\[{whitespace_pattern}\\]'
            allow_empty = '?' if int(instance.get('minItems', 0)) == 0 else ''
            if 'items' in instance:
                items_regex = to_regex(resolver, instance['items'], whitespace_pattern)
                return f'\\[{whitespace_pattern}(({items_regex})(,{whitespace_pattern}({items_regex})){num_repeats}){allow_empty}{whitespace_pattern}\\]'
            else:
                types = [{'type': 'boolean'}, {'type': 'null'}, {'type': 'number'}, {'type': 'integer'}, {'type': 'string'}]
                regexes = [to_regex(resolver, t, whitespace_pattern) for t in types]
                return f'\\[{whitespace_pattern}({'|'.join(regexes)})(,{whitespace_pattern}({'|'.join(regexes)})){num_repeats}){allow_empty}{whitespace_pattern}\\]'
        elif instance_type == 'object':
            num_repeats = _get_num_items_pattern(instance.get('minProperties'), instance.get('maxProperties'), whitespace_pattern)
            if num_repeats is None:
                return f'\\{{{whitespace_pattern}\\}}'
            allow_empty = '?' if int(instance.get('minProperties', 0)) == 0 else ''
            value_pattern = to_regex(resolver, instance['additionalProperties'], whitespace_pattern)
            key_value_pattern = f'{STRING}{whitespace_pattern}:{whitespace_pattern}{value_pattern}'
            key_value_successor_pattern = f'{whitespace_pattern},{whitespace_pattern}{key_value_pattern}'
            multiple_key_value_pattern = f'({key_value_pattern}({key_value_successor_pattern}){num_repeats}){allow_empty}'
            return '\\{' + whitespace_pattern + multiple_key_value_pattern + whitespace_pattern + '\\}'
        elif instance_type == 'boolean':
            return type_to_regex['boolean']
        elif instance_type == 'null':
            return type_to_regex['null']
        elif isinstance(instance_type, list):
            regexes = [to_regex(resolver, {'type': t}, whitespace_pattern) for t in instance_type if t != 'object']
            return f'({'|'.join(regexes)})'
    raise NotImplementedError(f'Could not translate the instance {instance} to a\n    regular expression. Make sure it is valid to the JSON Schema specification. If\n    it is, please open an issue on the Outlines repository')