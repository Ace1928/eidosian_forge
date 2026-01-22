import importlib
import locale
import os
import platform
import struct
import subprocess
import sys
def show_versions(file=sys.stdout):
    """print the versions of xarray and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    sys_info = get_sys_info()
    try:
        sys_info.extend(netcdf_and_hdf5_versions())
    except Exception as e:
        print(f'Error collecting netcdf / hdf5 version: {e}')
    deps = [('xarray', lambda mod: mod.__version__), ('pandas', lambda mod: mod.__version__), ('numpy', lambda mod: mod.__version__), ('scipy', lambda mod: mod.__version__), ('netCDF4', lambda mod: mod.__version__), ('pydap', lambda mod: mod.__version__), ('h5netcdf', lambda mod: mod.__version__), ('h5py', lambda mod: mod.__version__), ('Nio', lambda mod: mod.__version__), ('zarr', lambda mod: mod.__version__), ('cftime', lambda mod: mod.__version__), ('nc_time_axis', lambda mod: mod.__version__), ('iris', lambda mod: mod.__version__), ('bottleneck', lambda mod: mod.__version__), ('dask', lambda mod: mod.__version__), ('distributed', lambda mod: mod.__version__), ('matplotlib', lambda mod: mod.__version__), ('cartopy', lambda mod: mod.__version__), ('seaborn', lambda mod: mod.__version__), ('numbagg', lambda mod: mod.__version__), ('fsspec', lambda mod: mod.__version__), ('cupy', lambda mod: mod.__version__), ('pint', lambda mod: mod.__version__), ('sparse', lambda mod: mod.__version__), ('flox', lambda mod: mod.__version__), ('numpy_groupies', lambda mod: mod.__version__), ('setuptools', lambda mod: mod.__version__), ('pip', lambda mod: mod.__version__), ('conda', lambda mod: mod.__version__), ('pytest', lambda mod: mod.__version__), ('mypy', lambda mod: importlib.metadata.version(mod.__name__)), ('IPython', lambda mod: mod.__version__), ('sphinx', lambda mod: mod.__version__)]
    deps_blob = []
    for modname, ver_f in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
        except Exception:
            deps_blob.append((modname, None))
        else:
            try:
                ver = ver_f(mod)
                deps_blob.append((modname, ver))
            except Exception:
                deps_blob.append((modname, 'installed'))
    print('\nINSTALLED VERSIONS', file=file)
    print('------------------', file=file)
    for k, stat in sys_info:
        print(f'{k}: {stat}', file=file)
    print('', file=file)
    for k, stat in deps_blob:
        print(f'{k}: {stat}', file=file)