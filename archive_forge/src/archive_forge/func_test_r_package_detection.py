from jupyter_lsp.specs.r_languageserver import RLanguageServer
from jupyter_lsp.specs.utils import PythonModuleSpec
def test_r_package_detection():
    with_installed_server = RLanguageServer()
    assert with_installed_server.is_installed(mgr=None) is True

    class NonInstalledRServer(RLanguageServer):
        package = 'languageserver-fork'
    non_installed_server = NonInstalledRServer()
    assert non_installed_server.is_installed(mgr=None) is False