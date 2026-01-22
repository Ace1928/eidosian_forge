import os, subprocess, json
def get_setupcfg_version():
    """As get_setup_version(), but configure via setup.cfg.

    If your project uses setup.cfg to configure setuptools, and hence has
    at least a "name" key in the [metadata] section, you can
    set the version as follows:
    ```
    [metadata]
    name = mypackage
    version = attr: autover.version.get_setup_version2
    ```

    If the repository name is different from the package name, specify
    `reponame` as a [tool:autover] option:
    ```
    [tool:autover]
    reponame = mypackage
    ```

    To ensure git information is included in a git archive, add
    setup.cfg to .gitattributes (in addition to __init__):
    ```
    __init__.py export-subst
    setup.cfg export-subst
    ```

    Then add the following to setup.cfg:
    ```
    [tool:autover.configparser_workaround.archive_commit=$Format:%h$]
    ```

    The above being a section heading rather than just a key is
    because setuptools requires % to be escaped with %, or it can't
    parse setup.cfg...but then git export-subst would not work.

    """
    try:
        import configparser
    except ImportError:
        import ConfigParser as configparser
    import re
    cfg = 'setup.cfg'
    autover_section = 'tool:autover'
    config = configparser.ConfigParser()
    config.read(cfg)
    pkgname = config.get('metadata', 'name')
    reponame = config.get(autover_section, 'reponame', vars={'reponame': pkgname}) if autover_section in config.sections() else pkgname
    archive_commit = None
    archive_commit_key = autover_section + '.configparser_workaround.archive_commit'
    for section in config.sections():
        if section.startswith(archive_commit_key):
            archive_commit = re.match('.*=\\s*(\\S*)\\s*', section).group(1)
    return get_setup_version(cfg, reponame=reponame, pkgname=pkgname, archive_commit=archive_commit)