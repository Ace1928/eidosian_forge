"""
Dependency validation for version compatibility across components.

Precision-guided dependency graph analysis with strict validation
for ensuring consistent versioning across the Eidosian ecosystem.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple

from ..operations.compare import is_compatible
from ..protocols.interfaces import IVersioned

logger = logging.getLogger("forge.version")


class DependencyValidator:
    """
    Validate dependencies for version compatibility with structural integrity.

    The validator enforces dependency contracts between components, ensuring
    version compatibility and generating upgrade plans with topological awareness.
    """

    def __init__(self, components: Optional[Dict[str, IVersioned]] = None):
        """Initialize the validator with optional component dictionary"""
        self._components = components or {}
        self._dependencies: Dict[str, Set[str]] = {}

    def register_component(self, name: str, component: IVersioned) -> None:
        """Register a component for validation"""
        self._components[name] = component
        logger.debug(f"Registered component {name} v{component.version}")

    def register_dependency(self, dependent: str, dependency: str) -> None:
        """Register a dependency relationship between components"""
        if dependent not in self._dependencies:
            self._dependencies[dependent] = set()
        self._dependencies[dependent].add(dependency)
        logger.debug(f"Registered dependency: {dependent} → {dependency}")

    def validate_dependency_graph(self) -> Tuple[bool, List[str]]:
        """
        Validate the entire dependency graph, returning success and error messages.

        Returns a tuple of (is_valid, error_messages).
        """
        if not self._components or not self._dependencies:
            return True, []  # Empty graph is valid by definition

        errors: List[str] = []

        for dependent, dependencies in self._dependencies.items():
            if dependent not in self._components:
                errors.append(f"Component '{dependent}' is not registered but has dependencies")
                continue

            dependent_comp = self._components[dependent]
            dependent_min_version = self._get_min_version(dependent_comp)

            for dependency in dependencies:
                if dependency not in self._components:
                    errors.append(f"Dependency '{dependency}' is not registered but is required by '{dependent}'")
                    continue

                # Get component instances
                dependency_comp = self._components[dependency]
                dependency_version = str(dependency_comp.version)

                # Check compatibility using the ecosystem's compatibility function
                if not is_compatible(dependency_version, dependent_min_version):
                    errors.append(
                        f"Component '{dependent}' v{dependent_comp.version} is incompatible with "
                        f"its dependency '{dependency}' v{dependency_comp.version} "
                        f"(requires ≥ {dependent_min_version})"
                    )

        return len(errors) == 0, errors

    def get_upgrade_plan(self, target_versions: Dict[str, str]) -> Dict[str, str]:
        """
        Get an ordered upgrade plan to reach target versions.

        Uses topological sorting to determine the correct upgrade order that
        respects all dependency relationships.

        Returns a dictionary mapping component names to versions in the order
        they should be upgraded.
        """
        if not self._components or not self._dependencies:
            return {}

        # Build dependency graph
        graph: Dict[str, Set[str]] = {}
        for dependent, dependencies in self._dependencies.items():
            if dependent not in graph:
                graph[dependent] = set()
            for dependency in dependencies:
                graph[dependent].add(dependency)

        # Add components without dependencies
        for component in self._components:
            if component not in graph:
                graph[component] = set()

        # Topological sort - resolve dependency order
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        order: List[str] = []

        def visit(node: str) -> None:
            """Recursive topological sort visitor with cycle detection"""
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node in visited:
                return

            temp_visited.add(node)

            # Visit all dependencies first
            for dependency in graph.get(node, set()):
                visit(dependency)

            temp_visited.remove(node)
            visited.add(node)
            order.append(node)

        # Visit all nodes in the graph
        for node in graph:
            if node not in visited:
                visit(node)

        # Create upgrade plan in correct order
        plan: Dict[str, str] = {}
        for component in order:
            if component in target_versions:
                plan[component] = target_versions[component]

        return plan

    def find_compatible_version(self, component: str, dependency: str) -> Optional[str]:
        """
        Find a compatible version of a dependency for the given component.

        Returns the compatible version string or None if none found.
        """
        if component not in self._components or dependency not in self._components:
            return None

        component_comp = self._components[component]
        dependency_comp = self._components[dependency]

        component_min_version = self._get_min_version(component_comp)
        dependency_version = str(dependency_comp.version)

        # Check if current versions are compatible
        if is_compatible(dependency_version, component_min_version):
            return dependency_version

        # Future enhancement: search registry for other available versions
        return None

    def _get_min_version(self, component: IVersioned) -> str:
        """
        Extract minimum version requirement from a component.

        Handles various component structures gracefully.
        """
        # Try the most common attribute patterns
        min_version = getattr(component, 'min_version', None)
        if min_version is not None:
            return str(min_version)

        # Check version_info dictionary if available
        version_info = getattr(component, 'version_info', None)
        if version_info is not None:
            # Use dict access pattern safely
            if hasattr(version_info, 'get'):
                min_ver = version_info.get('min_version')
                if min_ver is not None:
                    return str(min_ver)

        # Default to 0.1.0 when no constraints are specified
        return "0.1.0"
