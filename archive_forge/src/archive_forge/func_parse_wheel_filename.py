import re
from typing import FrozenSet, NewType, Tuple, Union, cast
from .tags import Tag, parse_tag
from .version import InvalidVersion, Version
def parse_wheel_filename(filename: str) -> Tuple[NormalizedName, Version, BuildTag, FrozenSet[Tag]]:
    if not filename.endswith('.whl'):
        raise InvalidWheelFilename(f"Invalid wheel filename (extension must be '.whl'): {filename}")
    filename = filename[:-4]
    dashes = filename.count('-')
    if dashes not in (4, 5):
        raise InvalidWheelFilename(f'Invalid wheel filename (wrong number of parts): {filename}')
    parts = filename.split('-', dashes - 2)
    name_part = parts[0]
    if '__' in name_part or re.match('^[\\w\\d._]*$', name_part, re.UNICODE) is None:
        raise InvalidWheelFilename(f'Invalid project name: {filename}')
    name = canonicalize_name(name_part)
    version = Version(parts[1])
    if dashes == 5:
        build_part = parts[2]
        build_match = _build_tag_regex.match(build_part)
        if build_match is None:
            raise InvalidWheelFilename(f"Invalid build number: {build_part} in '{filename}'")
        build = cast(BuildTag, (int(build_match.group(1)), build_match.group(2)))
    else:
        build = ()
    tags = parse_tag(parts[-1])
    return (name, version, build, tags)