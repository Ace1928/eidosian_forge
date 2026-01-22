from jupyter_lsp.specs.r_languageserver import RLanguageServer
from jupyter_lsp.specs.utils import PythonModuleSpec
def test_missing_python_module_spec():
    """Prevent failure in module detection raising error"""

    class NonInstalledPythonServer(PythonModuleSpec):
        python_module = 'not_installed_python_module'
        key = 'a_module'
    not_installed_server = NonInstalledPythonServer()
    assert not_installed_server.is_installed(mgr=None) is False
    assert 'languages' in not_installed_server(mgr=None)['a_module']