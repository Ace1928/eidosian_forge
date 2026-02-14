Build Dependencies
==================

Build dependencies for Python-dependent packages must be declared for
every Python version that the package is built for.

The ``python3-all-dev`` should be used when building extensions for any
or all Python 3 versions.
The ``python-all-dev`` should be used when building extensions for any
or all Python 2 versions.
To build for a specific version or versions, declare ``Build-Depends``
on :samp:`python{X}.{Y}-dev`.

Some applications and pure Python modules may be able to avoid
dependency on the ``-dev`` packages, and declare ``Build-Depends`` on
the runtime environment only
(``python3``, ``python3-all``, ``python2``, ``python2-all``).
A package that does not require the ``-dev`` packages must not declare
``Build-Depends`` on them.

Declare ``Build-Depends`` on at least::

    Build-Depends: python2.7
    Build-Depends: python2.6 (>= 2.6-1)
    Build-Depends: python (>= 2.6.6-9)
    Build-Depends: python-all

    Build-Depends: python2.7-dev
    Build-Depends: python3.5-dev (>= 3.5.1-1)
    Build-Depends: python-dev (>= 2.6.6-9)
    Build-Depends: python-all-dev
    Build-Depends: python3-all-dev (>= 3.2)
