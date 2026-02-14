Upgrade Procedure
=================

This section describes the procedure for the upgrade when the
default Python version is changed in the Debian ``unstable`` release,
requiring recompilation of many Python-related packages.

#. Selected pre-releases and release candidates of new Python versions
   are uploaded to Debian ``experimental`` to support pre-transition
   work and testing.

#. Application and module maintainers make sourceful changes where
   needed to prepare for the new Python version when needed.

#. Have a long and heated discussion.

#. The Debian Python maintainer and module/application maintainers
   discuss the readiness for a new default Debian Python version and
   associated packaging/policy changes.
   Once there is some consensus, the Python maintainer announces the
   upgrade and uploads to ``unstable``.

#. Upload of the Python core meta-packages ``python3``,
   ``python3-dev``, ``python3-doc`` and several
   :samp:`python3-{module}`, depending on the new :samp:`python3.{Y}`,
   :samp:`python3.{Y}-dev` and so on.

#. The Debian release team schedules rebuilds for packages that may
   need it.
   Packages that require additional manual work get updated and
   uploaded.

The necessary package builds are typically done in three phases in order
to keep transitions as smooth as possible.
For Python 3, there is no general need to update architecture all
packages for a new Python 3 version. Only architecture any packages need
to be rebuilt.

#. The new Python 3 version is added to supported versions and packages
   that support multiple Python 3 versions are binNMUed.
   They now support both the new and older Python 3 versions.
   This requires transition assistance from the release team in the form
   of a transition tracker and binNMU scheduling, but is not a
   transition that can cause entanglements with other transitions in
   Debian.

#. Once the default Python 3 version is changed, binNMUs are done for
   packages that only support one Python 3 version.
   Some transient uninstallability is unavoidable.
   This is a transition that can entangle other transitions in Debian
   and requires more careful coordination with the release team.

#. After the old Python 3 version is dropped from supported versions
   then packages with multi-version support are binNMUed again to remove
   support for the old Python 3 version.
   This is not a true transition and only needs a tracker and binNMU
   scheduling.
