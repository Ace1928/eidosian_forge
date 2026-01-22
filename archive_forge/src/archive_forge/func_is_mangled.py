import re
def is_mangled(name: str) -> bool:
    return bool(re.match('<torch_package_\\d+>', name))