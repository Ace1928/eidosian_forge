Python Programs
===============

Interpreter directive ("Shebang")
---------------------------------

Executables written for interpretation by Python must use an appropriate
interpreter directive, or "shebang", as the first line of the program.
This line should be of the form :samp:`#!{interpreter_location}`.
See :ref:`interpreter-name` for the interpreter name to use.

As noted in :ref:`interpreter-location`, the form :samp:`#!/usr/bin/env
{interpreter_name}` is deprecated.

Programs using the default Python
---------------------------------

A package that installs a program that can be run by any version of
Python 3 must declare a dependency on ``python3``, with a versioned
dependency if necessary.

A package that installs a program that can be run by any version of
Python 2 must declare a dependency on ``python2``, with a versioned
dependency if necessary.

If the program needs the public Python module ``foo``, the package must
depend on the binary package that installs the ``foo`` module.
See :ref:`module-package-names` for the naming of packages that install
public Python modules.

Programs Shipping Private Modules
---------------------------------

A program that specifies ``python3`` as its interpreter may require its
own private Python modules.
These modules should be installed in :file:`/usr/share/{module}`, or
:file:`/usr/lib/{module}` if the modules are architecture-dependent
(e.g.  extensions).

The rules explained in :ref:`module-byte-compilation` apply to those
private modules: the byte-compiled modules must not be shipped with the
binary package, they should be generated in the package's post-install
script using the current default Python version, and removed in the
pre-remove script.
Modules should be byte-compiled using the current default Python
version.

Programs that have private compiled extensions must either handle
multiple version support themselves, or declare a tight dependency on
the current Python version (e.g. ``Depends: python3 (>= 3.5),
python3 (<< 3.6)``.

Programs Using a Particular Python Version
------------------------------------------

A program which requires a specific minor version of Python must specify
the versioned interpreter :samp:`python{X}.{Y}`.
The package that installs the programs must also specify a dependency on
:samp:`python{X}.{Y}` and on any packages that install necessary
modules.

The notes on installation directories and byte-compilation for programs
that support any version of Python also apply to programs supporting
only a single Python version.
Modules to be byte-compiled should use the same Python version as the
package itself.
