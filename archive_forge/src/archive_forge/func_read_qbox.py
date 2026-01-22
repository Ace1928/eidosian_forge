from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import reader
import re
import xml.etree.ElementTree as ET
@reader
def read_qbox(f, index=-1):
    """Read data from QBox output file

    Inputs:
        f - str or fileobj, path to file or file object to read from
        index - int or slice, which frames to return

    Returns:
        list of Atoms or atoms, requested frame(s)
    """
    version = None
    for line in f:
        if '<release>' in line:
            version = ET.fromstring(line)
            break
    if version is None:
        raise Exception('Parse Error: Version not found')
    is_qball = 'qb@LL' in version.text or 'qball' in version.text
    species = dict()
    if is_qball:
        species_data = []
        for line in f:
            if '<run' in line:
                break
            species_data.append(line)
        species_data = '\n'.join(species_data)
        symbols = re.findall('symbol_ = ([A-Z][a-z]?)', species_data)
        masses = re.findall('mass_ = ([0-9.]+)', species_data)
        names = re.findall('name_ = ([a-z]+)', species_data)
        numbers = re.findall('atomic_number_ = ([0-9]+)', species_data)
        for name, symbol, mass, number in zip(names, symbols, masses, numbers):
            spec_data = dict(symbol=symbol, mass=float(mass), number=float(number))
            species[name] = spec_data
    else:
        species_blocks = _find_blocks(f, 'species', '<cmd>run')
        for spec in species_blocks:
            name = spec.get('name')
            spec_data = dict(symbol=spec.find('symbol').text, mass=float(spec.find('mass').text), number=int(spec.find('atomic_number').text))
            species[name] = spec_data
    frames = _find_blocks(f, 'iteration', None)
    if isinstance(index, int):
        return _parse_frame(frames[index], species)
    else:
        return [_parse_frame(frame, species) for frame in frames[index]]