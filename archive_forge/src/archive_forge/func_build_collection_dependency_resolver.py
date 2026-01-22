from __future__ import (absolute_import, division, print_function)
import typing as t
from ansible.galaxy.collection.galaxy_api_proxy import MultiGalaxyAPIProxy
from ansible.galaxy.dependency_resolution.providers import CollectionDependencyProvider
from ansible.galaxy.dependency_resolution.reporters import CollectionDependencyReporter
from ansible.galaxy.dependency_resolution.resolvers import CollectionDependencyResolver
def build_collection_dependency_resolver(galaxy_apis, concrete_artifacts_manager, preferred_candidates=None, with_deps=True, with_pre_releases=False, upgrade=False, include_signatures=True, offline=False):
    """Return a collection dependency resolver.

    The returned instance will have a ``resolve()`` method for
    further consumption.
    """
    return CollectionDependencyResolver(CollectionDependencyProvider(apis=MultiGalaxyAPIProxy(galaxy_apis, concrete_artifacts_manager, offline=offline), concrete_artifacts_manager=concrete_artifacts_manager, preferred_candidates=preferred_candidates, with_deps=with_deps, with_pre_releases=with_pre_releases, upgrade=upgrade, include_signatures=include_signatures), CollectionDependencyReporter())