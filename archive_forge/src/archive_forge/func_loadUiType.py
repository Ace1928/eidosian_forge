from .Compiler import indenter, compiler
from .objcreator import widgetPluginPath
def loadUiType(uifile, from_imports=False, resource_suffix='_rc', import_from='.'):
    """loadUiType(uifile, from_imports=False, resource_suffix='_rc', import_from='.') -> (form class, base class)

    Load a Qt Designer .ui file and return the generated form class and the Qt
    base class.

    uifile is a file name or file-like object containing the .ui file.
    from_imports is optionally set to generate relative import statements.  At
    the moment this only applies to the import of resource modules.
    resource_suffix is the suffix appended to the basename of any resource file
    specified in the .ui file to create the name of the Python module generated
    from the resource file by pyrcc4.  The default is '_rc', i.e. if the .ui
    file specified a resource file called foo.qrc then the corresponding Python
    module is foo_rc.
    import_from is optionally set to the package used for relative import
    statements.  The default is ``'.'``.
    """
    import sys
    from PyQt5 import QtWidgets
    if sys.hexversion >= 50331648:
        from .port_v3.string_io import StringIO
    else:
        from .port_v2.string_io import StringIO
    code_string = StringIO()
    winfo = compiler.UICompiler().compileUi(uifile, code_string, from_imports, resource_suffix, import_from)
    ui_globals = {}
    exec(code_string.getvalue(), ui_globals)
    uiclass = winfo['uiclass']
    baseclass = winfo['baseclass']
    ui_base = ui_globals.get(baseclass)
    if ui_base is None:
        ui_base = getattr(QtWidgets, baseclass)
    return (ui_globals[uiclass], ui_base)