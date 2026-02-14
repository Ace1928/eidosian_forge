Completing the move to Python 3
===============================

Debian has previously supported two Python stacks, one for Python 3 and
one for Python 2.
The goal for Debian is to reduce this to one stack, dropping the Python
2 stack and interpreter for the Bullseye release.

:pep:`404` states that no more major Python 2 releases are planned,
although the latest released minor version 2.7 will see some extended
support, documented in :pep:`466`.

Packages in Debian should use Python 3.
New packages must use Python 3 from the initial upload, new upstream
versions for existing packages must use Python 3.
If Python 2 is still supported in Bullseye, selected packages may
continue using Python 2 until Python 3 support is available for those
packages.
Please discuss all use of Python 2 on the debian-python mailing list
before uploading.

#. Applications should use Python 3 and must not be packaged for Python
   2 as well.
   If an application supports only Python 2, the application may need
   to be removed from Debian so that it does not block removal of other
   Python 2 packages.

#. Python libraries need to support Python 3 and new versions must be
   packaged for Python 3.
   Existing Python 2 libraries must not be dropped before the last
   reverse dependency is removed.
   New Python 2 libraries must not be introduced.

#. Python 3 should be used for the packaging if the packaging scripts
   use Python.

Removal of the unversioned packages
-----------------------------------

Starting with the Debian 11 release (bullseye), the binary packages
``python``, ``python-minimal``, ``python-dev``, ``python-dbg`` and
``python-doc`` are removed.  No package in the archive must use any of
these packages as build dependencies, dependencies, recommendations or
suggestions.


Unversioned python commands
---------------------------

For the Debian 11 release (bullseye), the :file:`/usr/bin/python`
command is provided in the ``python-is-python2`` package (pointing to
:file:`/usr/bin/python2`).  The :file:`/usr/bin/python-config` and
:file:`/usr/bin/pydoc` commands are provided in the
``python-dev-is-python2`` package.  These package are not installed by
default for new installations, but only for upgrades from the Debian
10 release (buster).  These packages should be removed after an
upgrade.  These packages will not be part of the Debian 12 release
(bookworm).

The packages ``python-is-python3`` and ``python-dev-is-python3``
provide the :file:`/usr/bin/python`, :file:`/usr/bin/python-config`
and :file:`/usr/bin/pydoc` commands pointing to Python3.  These
packages can be installed by developers and users to use the
unversioned commands.  NOTE: Locally installed software not yet ported
to Python3 is likely to break when installing these packages.

The packages ``python-is-python3``, ``python-dev-is-python3``,
``python-is-python2`` and ``python-dev-is-python2`` must not be used
as build dependencies, dependencies, recommendations or suggestions.
