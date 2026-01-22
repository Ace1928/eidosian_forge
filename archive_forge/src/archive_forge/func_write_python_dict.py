import json
import xml.etree.ElementTree as ET
from langcodes.util import data_filename
from langcodes.registry_parser import parse_registry
def write_python_dict(outfile, name, d):
    print(f'{name} = {{', file=outfile)
    for key in sorted(d):
        value = d[key]
        print(f'    {key!r}: {value!r},', file=outfile)
    print('}', file=outfile)