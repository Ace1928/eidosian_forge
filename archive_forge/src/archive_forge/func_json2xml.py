import sys
import os
import argparse
import logging
import pathlib
from urllib.error import URLError
import xmlschema
from xmlschema import XMLSchema, XMLSchema11, iter_errors, to_json, from_json, etree_tostring
from xmlschema.exceptions import XMLSchemaValueError
def json2xml():
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, add_help=True, description='encode a set of JSON files to XML.')
    parser.usage = "%(prog)s [OPTION]... [FILE]...\nTry '%(prog)s --help' for more information."
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='increase output verbosity.')
    parser.add_argument('--schema', type=str, metavar='PATH', help='path or URL to an XSD schema.')
    parser.add_argument('--version', type=xsd_version_number, default='1.0', help='XSD schema validator to use (default is 1.0).')
    parser.add_argument('-L', dest='locations', nargs=2, type=str, action='append', metavar='URI/URL', help='schema location hint overrides.')
    parser.add_argument('--converter', type=str, metavar='NAME', help='use a different XML to JSON convention instead of the default converter. Option value can be one of {!r}.'.format(tuple(CONVERTERS_MAP)))
    parser.add_argument('--indent', type=int, default=4, help='indentation for XML output (default is 4 spaces)')
    parser.add_argument('-o', '--output', type=str, default='.', help='where to write the encoded XML files, current dir by default.')
    parser.add_argument('-f', '--force', action='store_true', default=False, help='do not prompt before overwriting')
    parser.add_argument('files', metavar='[JSON_FILE ...]', nargs='+', help='JSON files to be encoded to XML.')
    args = parser.parse_args()
    loglevel = get_loglevel(args.verbosity)
    schema_class = XMLSchema if args.version == '1.0' else XMLSchema11
    converter = get_converter(args.converter)
    schema = schema_class(args.schema, locations=args.locations, loglevel=loglevel)
    base_path = pathlib.Path(args.output)
    if not base_path.exists():
        base_path.mkdir()
    elif not base_path.is_dir():
        raise XMLSchemaValueError('{!r} is not a directory'.format(str(base_path)))
    tot_errors = 0
    for json_path in map(pathlib.Path, args.files):
        xml_path = base_path.joinpath(json_path.name).with_suffix('.xml')
        if xml_path.exists() and (not args.force):
            print('skip {}: the destination file exists!'.format(str(xml_path)))
            continue
        with open(str(json_path)) as fp:
            try:
                root, errors = from_json(source=fp, schema=schema, converter=converter, validation='lax', indent=args.indent)
            except (xmlschema.XMLSchemaException, URLError) as err:
                tot_errors += 1
                print('error with {}: {}'.format(str(xml_path), str(err)))
                continue
            else:
                if not errors:
                    print('{} converted to {}'.format(str(json_path), str(xml_path)))
                else:
                    tot_errors += len(errors)
                    print('{} converted to {} with {} errors'.format(str(json_path), str(xml_path), len(errors)))
        with open(str(xml_path), 'w') as fp:
            fp.write(etree_tostring(root))
    sys.exit(tot_errors)