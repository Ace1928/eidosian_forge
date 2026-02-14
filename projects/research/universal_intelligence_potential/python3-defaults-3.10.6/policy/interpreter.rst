Python Packaging
================

Versions
--------
At any given time, the binary package ``python3`` will represent the
current default Debian Python 3 version; the binary package ``python``
will represent the current default Debian Python 2 version, for as long
as it exists.
As far as is reasonable, Python 3 and Python 2 should be treated as
separate runtime systems with minimal interdependencies.

In some cases, Python policy explicitly references Python helper tools.
For Debian Stretch, the ``dh-python`` package provides the only such
tools; earlier helpers have been removed from Debian.

It is a design goal to fully specify required interfaces and functions
in policy for Python 3 and to avoid enshrining specific implementation
details in policy.
Except as noted, policy for Python 2 is the same as Python 3 with the
exception of the different major version number as needed to distinguish
them.

The default Debian Python version, for each of Python 3 and Python 2,
should always be the latest stable upstream version that can be fully
integrated in Debian.

There may be newer supported or unsupported versions included in Debian
if they are not fully integrated for a particular release.

Apart from the default version, legacy versions of Python or beta
releases of future upstream versions may be included as well in Debian,
as long as they are needed by other packages, or as long as it seems
reasonable to provide them.

Note: For the scope of this document, a Python version is synonymous
with all micro versions within that minor version.
e.g. Python 3.5.0 and 3.5.1 are micro versions of the same Python
version 3.5, but Python 3.4 and 3.5 are indeed different versions.

For any version, the main binary package must be called
:samp:`python{X}.{Y}`.

The set of currently supported Python 3 versions can be found in
:file:`/usr/share/python3/debian_defaults`; the supported interface to
this information is through :file:`/usr/bin/py3versions`.
The set of currently supported Python 2 versions can be found in
:file:`/usr/share/python/debian_defaults`; the supported interface to
this information is :file:`/usr/bin/pyversions`.

These files are in Python ``configparser`` format.
They define (in the ``DEFAULT`` section) the following options:

``default-version``
    The name of the interpreter for the current default Debian Python.
``supported-versions``
    The set of interpreter names currently supported and for which
    modules should be built and byte-compiled.
    This includes ``default-version``.
``old-versions``
    The set of interpreter names which might still be on the system but
    for which modules should not be built.
``unsupported-versions``
    The set of interpreter names which should not be supported at all,
    that is modules should not be built or byte-compiled for these.
    This includes (is a superset of) ``old-versions``.

Newer versions might also appear in ``unsupported-versions`` before
being moved to ``supported-versions``.

.. _main-packages:

Main packages
-------------

For every Python version provided in Debian, the binary package
:samp:`python{X}.{Y}` shall provide a complete distribution for
*deployment* of Python scripts and applications.
The package must ensure that the binary :file:`/usr/bin/python{X}.{Y}`
is provided.

Installation of :samp:`python{X}.{Y}` shall provide the modules of the
upstream Python distribution with some exceptions.

Excluded are modules that cannot be included for licensing reasons, for
dependency tracking purposes (for example the GPL-licensed ``gdbm``
module), or that should not be included for packaging reasons (for
example the ``tk`` module which depends on Xorg and the ``venv`` module
which depends on wheels to bootstrap pip).
Modules that would interfere with system package management (for example
``ensurepip``, when used outside virtual environments) are modified to
print a message explaining the problem and recommending alternatives.

Excluded are modules that cannot be included for licensing reasons (for
example the ``profile`` module), for dependency tracking purposes (for
example the GPL-licensed ``gdbm`` module), or that should not be
included for packaging reasons (for example the ``tk`` module which
depends on Xorg).

Some tools and files for the *development* of Python modules are split
off in a separate binary package :samp:`python{X}.{Y}-dev`.

Modules only used for *building* of Python modules (e.g. ``distutils``
and ``lib2to3``) are split into separate packages.
The ``python3-venv`` binary package depends on these.

Documentation will be provided separately as well.

At any time, the ``python3`` binary package must ensure that
:file:`/usr/bin/python3` is provided, as a symlink to the current
:samp:`python3.{Y}` executable.
The package must depend on the :samp:`python3.{Y}` package that installs
the executable.

A ``python3-full`` binary package must ensure that the entire Python
standard library is available, including all modules split into separate
packages (but excluding modules excluded from Debian for licensing
reasons).
This package exists for the convenience of python developers, and must
not be used in dependencies, recommendations and build dependencies by
python module or application packages.

The version of the ``python3`` package must be greater than or equal to
:samp:`3.{Y}` and lower than :samp:`3.{Y+1}`.

The ``python`` and ``python-dbg`` binary packages are to be removed for
Bullseye.
If any Python 2 packages remain in Bullseye, these must depend on
``python2`` or ``python2-dbg``.
The ``python2`` package must depend on the :samp:`python2.{Y}` package
that installs the executable :file:`/usr/bin/python2`.
The version of the ``python2`` package must be greater than or equal to
:samp:`2.{Y}` and lower than :samp:`2.{Y+1}`.
The ``python2`` must not provide :file:`/usr/bin/python`.

For as long as it remains supported, the ``python`` binary package must
ensure that :file:`/usr/bin/python2` is provided, as a symlink to the
current :samp:`python2.{Y}` executable.
The package must depend on the :samp:`python2.{Y}` package that installs
the executable.

The ``python`` binary package must also ensure that
:file:`/usr/bin/python` is provided, as a symlink to the current
:samp:`python2.{Y}` executable.
See :pep:`394` for details.

The version of the ``python`` package must be greater than or equal to
:samp:`2.{Y}` and lower than :samp:`2.{Y+1}`.

Minimal packages
----------------

For every Python version provided in Debian, the binary package
:samp:`python{X}.{Y}-minimal` might exist and should not be depended
upon by other packages except the Python runtime packages themselves.

Python Interpreter
------------------

.. _interpreter-name:

Interpreter Name
^^^^^^^^^^^^^^^^

The different Python major versions require different interpreters
(see :ref:`main-packages`).

Python scripts that require the default Python 3 version should specify
``python3`` as the interpreter name.

Python scripts that require the default Python 2 version should specify
``python2`` as the interpreter name for as long as this remains supported.

Python scripts should not specify ``python`` as the interpreter name
even if they do not require any particular version of Python as the
script would stop working upon removal of the Python 2 stack.

Python scripts that only work with a specific Python minor version must
explicitly use the versioned interpreter name (:samp:`python{X}.{Y}`).

.. _interpreter-location:

Interpreter Location
^^^^^^^^^^^^^^^^^^^^

Python scripts should specify the Debian Python interpreter, to ensure
that the Debian Python installation is used and all dependencies on
additional Python modules are met.

The preferred specification for the Python 3 interpreter is
:file:`/usr/bin/python3` (or :file:`/usr/bin/python3.{Y}` if it requires
Python :samp:`3.{Y}`).

The preferred specification for the Python 2 interpreter is
:file:`/usr/bin/python2` (or :file:`/usr/bin/python2.{Y}` if it requires
Python :samp:`2.{Y}`).

Scripts requiring the default Python 2 version must not specify the
interpreter :file:`/usr/bin/python` as such scripts will fail when the
unversioned interpreter binary :file:`/usr/bin/python` is removed.

Maintainers should not override the Debian Python interpreter using
:samp:`/usr/bin/env {name}`.
This is not advisable as it bypasses Debian's dependency checking and
makes the package vulnerable to incomplete local installations of
Python.

.. _module-path:

Module Path
-----------

By default, Python modules are searched in the directories listed in the
:envvar:`PYTHONPATH` environment variable and in the ``sys.path`` Python
variable.
For all supported Debian releases, ``sys.path`` does not include a
:file:`/usr/lib/python{X}{Y}.zip` entry.

Directories with private Python modules must be absent from the ``sys.path``.

Public Python 3 modules must be installed in the system Python 3 modules
directory, :file:`/usr/lib/python3/dist-packages`.

Public Python 2 modules must be installed in the system Python 2 modules
directory :file:`/usr/lib/python2.{Y}/dist-packages`, where
:samp:`2.{Y}` is the Python 2 version.

A special directory is dedicated to public Python modules installed by
the local administrator, :file:`/usr/local/lib/python3/dist-packages`
for all Python 3 versions,
:file:`/usr/local/lib/python2.{Y}/dist-packages` for Python 2.

For local installation of Python modules by the system administrator,
special directories are reserved.
The directory :file:`/usr/local/lib/python3/site-packages` is in the
Python 3 runtime module search path.
The directory :file:`/usr/local/lib/python2.{Y}/site-packages` is in the
Python :samp:`2.{Y}` runtime module search path.

Additional information on appending site-specific paths to the module
search path is available in the official documentation of the ``site``
module.

Python modules which work with multiple supported Python 2 versions must
install to version-specific locations, for instance
:file:`/usr/lib/python2.6/dist-packages/foo.py` and
:file:`/usr/lib/python2.7/dist-packages/foo.py`.
These should point to a common file.

Architecture-independent public Python 3 modules must be installed to
:file:`/usr/lib/python3/dist-packages`.

Architecture-independent public Python 2 modules should be installed to
:file:`/usr/lib/python2.7/dist-packages`.
The historical location for this was :file:`/usr/share/pyshared`.
Since Python 2.7 is the last Python 2 version and the only supported
version in Wheezy and later releases, a version-specific location is
sufficient.

Hooks for updates to installed runtimes
---------------------------------------

The ``python3`` binary package has special hooks to allow other packages
to act upon updates to the installed runtimes.

This mechanism is required to handle changes of the default Python
runtime in some packages and to enable the Python packaging helpers.

There are three supported hook types which come in the form of scripts
which are invoked from the maintainer scripts of the Python runtime
packages when specific installations, removals, or upgrades occur.

#. :file:`/usr/share/python3/runtime.d/{*}.rtinstall`:
   These are called when a runtime is installed or becomes supported.
   The first argument is ``rtinstall``, the second argument is the
   affected runtime (for example :samp:`python{X}.{Y}`) and the third
   and fourth argument are the old and new version of this packaged
   runtime if this runtime was already installed but unsupported.

#. :file:`/usr/share/python3/runtime.d/{*}.rtremove`:
   These are called when a runtime is removed or stops being supported.
   The first argument is ``rtremove``, and the second argument is the
   affected runtime (for example :samp:`python{X}.{Y}`).

#. :file:`/usr/share/python3/runtime.d/{*}.rtupdate`:
   These are called when the default runtime changes.
   The first argument is either ``pre-rtupdate``, called before changing
   the default runtime, or ``rtupdate``, called when changing the
   default runtime, or ``post-rtupdate``, called immediately afterwards.
   The second argument is the old default runtime (for example
   :samp:`python{X}.{Y}`), and the third argument is the new default
   runtime (for example :samp:`python{X}.{Z}`).

Documentation
-------------

Python documentation is split out in separate binary packages
:samp:`python{X}.{Y}-doc`.

The binary package ``python3-doc`` will always provide the documentation
for the default Debian Python 3 version.
The binary package ``python2-doc`` will always provide the documentation
for the default Debian Python 2 version, for as long as that remains
supported.

TODO: Policy for documentation of third party packages.
