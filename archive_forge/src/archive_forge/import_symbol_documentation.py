from importlib import import_module
from traits.trait_base import xgetattr
 Import the symbol defined by the specified symbol path.

    Examples
    --------

    import_symbol('tarfile:TarFile') -> TarFile
    import_symbol('tarfile:TarFile.open') -> TarFile.open

    To allow compatibility with old-school traits symbol names we also allow
    all-dotted paths, but in this case you can only import top-level names
    from the module.

    import_symbol('tarfile.TarFile') -> TarFile

    