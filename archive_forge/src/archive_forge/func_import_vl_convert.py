from types import ModuleType
from packaging.version import Version
from importlib.metadata import version as importlib_version
def import_vl_convert() -> ModuleType:
    min_version = '1.3.0'
    try:
        version = importlib_version('vl-convert-python')
        if Version(version) < Version(min_version):
            raise RuntimeError(f'The vl-convert-python package must be version {min_version} or greater. Found version {version}')
        import vl_convert as vlc
        return vlc
    except ImportError as err:
        raise ImportError(f"""The vl-convert Vega-Lite compiler and file export feature requires\nversion {min_version} or greater of the 'vl-convert-python' package. \nThis can be installed with pip using:\n   pip install "vl-convert-python>={min_version}"\nor conda:\n   conda install -c conda-forge "vl-convert-python>={min_version}"\n\nImportError: {err.args[0]}""") from err