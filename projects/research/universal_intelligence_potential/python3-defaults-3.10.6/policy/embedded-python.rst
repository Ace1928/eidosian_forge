Programs Embedding Python
=========================

Building Embedded Programs
--------------------------

Any package that installs a program which embeds a Python interpreter
must declare ``Build-Depends`` on :samp:`python{X}.{Y}-dev`, where
:samp:`{X}.{Y}` is the Python version the program builds against.
It should be the current default Python version unless the program does
not work correctly with this version.

Embedded Python Dependencies
----------------------------

Dependencies for programs linking against the shared Python library will
be automatically created by ``dpkg-shlibdeps``.
The :samp:`libpython{X}.{Y}.so.{Z}` library the program is built against
is provided by the :samp:`python{X}.{Y}` package.
