Packaged Modules
================

The goal of these policies is to reduce the work necessary for Python
transitions.
Python modules are internally very dependent on a specific Python
version.
However, we want to automate recompiling modules when possible, either
during the upgrade itself (re-compiling bytecode files :file:`{*}.pyc`
and :file:`{*}.pyo`) or shortly thereafter with automated rebuilds (to
handle C extensions).
These policies encourage automated dependency generation and loose
version bounds whenever possible.

Types of Python Modules
-----------------------

There are two kinds of Python modules, "pure" Python modules, and
extension modules.
Pure Python modules are Python source code that generally works across
many versions of Python.
Extensions are C code compiled and linked against a specific version of
the Python runtime, and so can only be used by one version of Python.

Debian Python does not link extensions to ``libpython`` (as is done in
some operating systems).
Symbols are resolved by :file:`/usr/bin/python{X}.{Y}` which is not
linked to ``libpython``.

Python packages are a way of structuring Python’s module namespace by
using "dotted module names".
See `Python's glossary
<https://docs.python.org/3/glossary.html#term-package>`_ for details on
how packages are defined in Python terms (a package in the Python sense
is unrelated to a Debian package).
Python packages must be packaged into the same directory (as done by
upstream).
Splitting components of a package across directories changes the import
order and may confuse documentation tools and IDEs.

There are two ways to distribute Python modules.
Public modules are installed in a public directory as listed in
:ref:`module-path`.
They are accessible to any program.
Private modules are installed in a private directory such as
:file:`/usr/share/{package-name}` or :file:`/usr/lib/{package-name}`.
They are generally only accessible to a specific program or suite of
programs included in the same package.

Wheels
------

:pep:`427` defines a built-package format called "wheels", which is a
Zip format archive containing Python code and a :file:`{*}.dist-info`
metadata directory, in a single file named with the ``.whl`` suffix.
As Zip files, wheels containing pure Python can be put on ``sys.path``
and modules in the wheel can be imported directly by Python's ``import``
statement.
(Importing extension modules from wheels is not yet supported as of
Python 3.4.)

Except as described below, packages must not build or provide wheels.
They are redundant to the established way of providing Python libraries
to Debian users, take no advantage of distro-based tools, and are less
convenient to use.
E.g. they must be explicitly added to ``sys.path``, cannot be easily
grepped, and stack traces through Zip files are more difficult to debug.

A very limited set of wheel packages are available in the archive, but
these support the narrow purpose of enabling the ``pip``,
``virtualenv``, and ``pyvenv`` tools in a Debian policy compliant way.
These packages build their own dependent wheels through the use of the
``dirtbike`` "rewheeling" tool, which takes installed Debian packages
and turns them back into wheels.
Only universal wheels (i.e. pure-Python, Python 3 and 2 compatible
packages) are supported, with the exception of wheels of packages that
no longer support Python 2.
Wheels built for these packages are not required to be universal.
Since only the programs that require wheels need build them, only they
may provide ``-whl`` packages, e.g. ``python3-pip-whl``.

When these binary packages are installed, :file:`{*}.whl` files must be
placed in the :file:`/usr/share/python-wheels` directory.
The location inside a virtual environment will be rooted in the virtual
environment, instead of :file:`/usr`.

.. _module-package-names:

Module Package Names
--------------------

Public Python modules must be packaged separately by major Python
version, to preserve run time separation between Python 2 and Python 3.

Public Python 3 modules used by other packages must have their binary
package name prefixed with ``python3-``.
It is recommended to use this prefix for all packages with public
modules as they may be used by other packages in the future.

The binary package for module :samp:`{foo}` should preferably be named
:samp:`python3-{foo}`, if the module name allows.
This is not required if the binary package installs multiple modules, in
which case the maintainer shall choose the name of the module which best
represents the package.

For the purposes of package naming, the name that is used for a module
is the name that can be used with ``import``, which is not necessarily
the same as the name used in setuptools ``PKG-INFO`` and ``.egg-info``
files and directories.
For example, the module described in :samp:`pyxdg-{*}.egg-info` is used
via ``import xdg``, so its package name is ``python3-xdg`` and not
``python3-pyxdg``.

Some modules have names that contain underscores or capital letters,
which are not allowed in Debian package names.
The recommendation is to replace underscores with hyphen/minus and
capital letters with lower-case.
For example, the modules that can be used with ``import distro_info``
and ``import Xlib`` are packaged as ``python3-distro-info`` and
``python3-xlib`` respectively.

For subpackages such as :samp:`{foo.bar}`, the recommendation is to name
the binary package :samp:`python3-{foo.bar}`.

Such a package should support the current Debian Python version,
and more if possible (there are several tools to help implement
this, see :ref:`packaging-tools`).
For example, if Python 3.3, 3.4, and 3.5 are supported, the Python
statement

.. code-block:: python

    import foo

should import the module when the program interpreter is any of
:file:`/usr/bin/python3.3`, :file:`/usr/bin/python3.4`, and
:file:`/usr/bin/python3.5`.
This requirement also applies to extension modules; binaries for all the
supported Python versions should be included in a single package.

Packages intended for use with Django (``python3-django``) are installed
in the same namespace as other python packages for a variety of reasons.
Many such packages are named :samp:`django_{name}` upstream.
These are then packaged as :samp:`python3-django-{name}`.
This makes it clear that they are intended for use with Django and not
general purpose Python modules.
Debian maintainers are encouraged to work with their upstreams to
support consistent use of this approach.

If the documentation for a module :samp:`{foo}` provided in
:samp:`python3-{foo}` is large enough that a separate binary package for
documentation is desired, then the documentation package should
preferably be named :samp:`python-{foo}-doc` (and in particular, not
:samp:`python3-{foo}-doc`).

Specifying Supported Versions
-----------------------------

The :file:`debian/control` source paragraph may contain optional fields
to specify the versions of Python the package supports.

The optional ``X-Python3-Version`` field specifies the versions of
Python 3 supported.
When not specified, it defaults to all currently supported Python 3
versions.

Similarly, the optional fields ``X-Python-Version`` or
``XS-Python-Version`` were used to specify the versions of Python 2
supported by the source package.
They are obsolete and must be removed.

These fields are used by some packaging scripts to automatically
generate appropriate Depends and Provides lines.
The format of the field may be one of the following::

    X-Python3-Version: >= X.Y
    X-Python3-Version: >= A.B, << X.Y
    XS-Python-Version: A.B, X.Y

The keyword ``all`` is no longer to be used since using version numbers
is clearer than ``all`` and encodes more information.
The keyword ``all`` must be ignored for Python 3 versions.

A comma-separated list of multiple individual versions (e.g. ``3.3, 3.4,
3.5``) in ``XS-Python-Version`` will continue to be supported, but
is not recommended.

The use of multiple individual versions in ``X-Python-Version`` or
``X-Python3-Version`` is not supported for Wheezy and later releases.

The keyword ``current`` has been deprecated and must not be used.
It must be ignored for Python 3 versions.

The use of ``XB-Python-Version`` in the binary package paragraphs of
:file:`debian/control` file has been deprecated and should be removed in
the normal course of package updates.
It never achieved sufficient deployment to support its intended purpose
of managing Python transitions.
This purpose can be adequately accomplished by examining package
dependencies.

Dependencies
------------

Any package that installs modules for the default Python version (or
many versions including the default) as described in
:ref:`module-package-names`, must declare a dependency on the default
Python runtime package.
If it requires other modules to work, the package must declare
dependencies on the corresponding packaged modules.
The package must not declare dependency on any version-specific Python
runtime or module package.

For Python 3, the correct dependencies are :samp:`Depends:
python3 (>= 3.{Y})` and any corresponding :samp:`python3-{foo}`
packages.

If any Python 2 packages remain, the correct dependencies are
:samp:`Depends: python2 (>= 2.{Y})` and any corresponding
:samp:`python2-{foo}` packages.

Any package that installs Python modules or Python 3 binary extensions
must also declare a maximum version it supports as currently built.
This is accomplished by declaring a maximum version constraint strictly
less than one higher than the current maximum version, i.e.
:samp:`Depends: python3 (<< {X}.{Y})`.

Provides
--------

Binary packages that declare Provides dependencies of the form
:samp:`python{X}.{Y}-foo` were never supported for Python 3.
They should be removed in the normal course of package updates.
Future provision of values for the substitution variable
``python:Provides`` is not guaranteed.

.. _module-byte-compilation:

Modules Byte-Compilation
------------------------

If a binary package provides any binary-independent modules
(:file:`{foo}.py` files), the corresponding byte-compiled modules
(:file:`{foo}.pyc` files) and optimized modules (:file:`{foo}.pyo`
files) must not ship in the package.
Instead, they should be generated in the package's post-install script,
and removed in the package's pre-remove script.
The package's :file:`prerm` has to make sure that both :file:`{foo}.pyc`
and :file:`{foo}.pyo` are removed.

A binary package should only byte-compile the files which belong to the
package.

The file :file:`/etc/python/debian_config` allows configuration how
modules should be byte-compiled.
The post-install scripts should respect these settings.

Pure Python modules in private installation directories that are
byte-compiled with the default Python version must be forcefully
byte-compiled again when the default Python version changes.

Public Python extensions should be bin-NMUed.

Private Python extensions should be subject to binary NMUs every time
the default interpreter changes, unless the extension is updated through
a :file:`{*}.rtupdate` script.
