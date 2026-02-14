.. _packaging-tools:

Packaging Tools
===============

This section describes the various tools to help package Python programs
and modules for Debian.
Although none of these tools are mandatory, their use is strongly
encouraged, as the above policy has been designed with them in mind (and
vice versa).
This appendix is just an overview.
If you use these tools, you should read their full documentation.

distutils
---------

The standard Python ``distutils`` module has been modified in Debian to
change the default installation directory of public Python modules and
to add a new flag to the ``install`` command to override the default,
``--install-layout=``.

Public Python modules installed with a modified distutils default to
:file:`/usr/local/lib/python{X}.{Y}/dist-packages` for Python 2.6 and
later.
This directory is seen by the system-provided Python 2.6.

When using a local Python installation, the default is
:file:`/usr/local/lib/python{X}.{Y}/site-packages` which is only seen by
the local Python installation.

Using the ``--install-layout=deb`` flag to the ``install`` command of
:file:`setup.py` with a system-provided Python 2.6 or later versions,
Python modules will be installed to
:file:`/usr/lib/python{X}.{Y}/dist-packages` which is only seen by the
system-provided Python, not by a local installation.

setuptools
----------

The related Python ``setuptools`` module has been modified in Debian
along the same lines as ``distutils``.

Upstream focus on developments and improvements for Python packaging
tools has largely shifted away from ``distutils`` and to ``setuptools``.
They offer a similar API and at some point in the future, ``setuptools``
may fully replace ``distutils`` in Debian package builds.

dh-python
---------

``dh-python`` provides extensions for ``debhelper`` to make it easier to
package Python modules and extensions.
They calculate Python dependencies, add maintainer scripts to byte
compile files, etc.
Their use is not mandatory, but they are recommended by the Debian
Python maintainers.
See `man dh_python3 <https://manpages.debian.org/dh_python3>`_ for
details.

pybuild
-------

Pybuild is a Debian Python specific build system that invokes various
build systems for requested Python versions in order to build modules
and extensions.
It supports automatically building for multiple Python versions.

CDBS
----

The CDBS :file:`python-distutils.mk` class helps packaging of distutils
based Python packages.

python-support (removed)
------------------------

``python-support`` provided another way to manage Python modules.
It has been removed from Debian Stretch and later releases.

python-central (removed)
------------------------
``python-central`` provided another way to manage Python modules.
It has been removed from Debian Jessie and later releases.
