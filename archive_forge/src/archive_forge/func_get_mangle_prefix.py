import re
def get_mangle_prefix(name: str) -> str:
    return name.partition('.')[0] if is_mangled(name) else name