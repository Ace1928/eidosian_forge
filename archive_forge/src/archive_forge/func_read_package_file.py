from io import open
from pathlib import Path
def read_package_file(*path):
    """Return the content of a file from the itables package"""
    with open(find_package_file(*path), encoding='utf-8') as fp:
        return fp.read()