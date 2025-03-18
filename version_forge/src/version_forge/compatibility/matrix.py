"""
Compatibility matrix for tracking and enforcing cross-component version relationships.

Provides elegant tracking of component version compatibility across the Eidosian ecosystem
with both programmatic and visual representations for intelligent decision making.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, cast

from numpy._typing import NDArray

from ..operations.compare import is_compatible
from ..protocols.interfaces import IVersioned

# Type definitions for matplotlib (imported conditionally to maintain compatibility)
FloatingArray = NDArray[Any]  # Type alias for floating-point arrays

# Enhanced Protocol definitions for matplotlib classes
class FigureProto(Protocol):
    """Protocol defining minimal Figure interface."""
    def savefig(self, fname: str, **kwargs: Any) -> None: ...

class AxesProto(Protocol):
    """Protocol defining minimal Axes interface."""
    pass

class AxesImageProto(Protocol):
    """Protocol defining minimal AxesImage interface."""
    pass

class ColorbarProto(Protocol):
    """Protocol defining minimal Colorbar interface."""
    pass

# Protocol for matplotlib.pyplot interface - the genius is in flexibility
class PyplotProto(Protocol):
    """Protocol defining essential pyplot functionality with maximum compatibility."""
    def figure(self, *args: Any, **kwargs: Any) -> FigureProto: ...
    def matshow(self, *args: Any, **kwargs: Any) -> AxesImageProto: ...
    def colorbar(self, *args: Any, **kwargs: Any) -> ColorbarProto: ...
    def xticks(self, *args: Any, **kwargs: Any) -> Any: ...
    def yticks(self, *args: Any, **kwargs: Any) -> Any: ...
    def title(self, *args: Any, **kwargs: Any) -> Any: ...
    def tight_layout(self, *args: Any, **kwargs: Any) -> None: ...
    def savefig(self, *args: Any, **kwargs: Any) -> None: ...
    def show(self, *args: Any, **kwargs: Any) -> None: ...

logger = logging.getLogger("forge.version")

class CompatibilityMatrix:
    """
    Track and enforce cross-component version compatibility with structural precision.

    The matrix maintains bidirectional compatibility relationships between components,
    ensuring consistency and providing visualization tools to communicate complex
    dependency structures.
    """

    def __init__(self):
        """Initialize an empty compatibility matrix with perfect symmetry."""
        self._compatibility_map: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        self._versions: Dict[str, List[str]] = {}

    def register_component(self, name: str, component: IVersioned) -> None:
        """Register a component's version information with graceful fallbacks."""
        version_str = str(component.version)
        # Handle possible absence of min_version in the IVersioned protocol
        min_version = getattr(component, 'min_version', component.version)
        min_version_str = str(min_version)

        # Initialize data structures if needed
        if name not in self._versions:
            self._versions[name] = []
        if name not in self._compatibility_map:
            self._compatibility_map[name] = {}

        # Add version if not already registered
        if version_str not in self._versions[name]:
            self._versions[name].append(version_str)

        # Log registration
        logger.debug(f"Registered component {name} v{version_str} (min: v{min_version_str})")

    def register_compatibility(self, component1: str, version1: str,
                              component2: str, version2: str) -> None:
        """Register bidirectional compatibility between specific component versions."""
        # Initialize data structures if needed
        if component1 not in self._compatibility_map:
            self._compatibility_map[component1] = {}
        if component2 not in self._compatibility_map[component1]:
            self._compatibility_map[component1][component2] = []

        # Record bidirectional compatibility
        compat_pair = (version1, version2)
        if compat_pair not in self._compatibility_map[component1][component2]:
            self._compatibility_map[component1][component2].append(compat_pair)

        # Ensure reverse mapping exists - relationship symmetry is guaranteed
        if component2 not in self._compatibility_map:
            self._compatibility_map[component2] = {}
        if component1 not in self._compatibility_map[component2]:
            self._compatibility_map[component2][component1] = []

        # Record reverse compatibility
        reverse_pair = (version2, version1)
        if reverse_pair not in self._compatibility_map[component2][component1]:
            self._compatibility_map[component2][component1].append(reverse_pair)

        logger.debug(f"Registered compatibility: {component1} v{version1} ↔ {component2} v{version2}")

    def verify_compatibility(self, component1: str, version1: str,
                           component2: str, version2: str) -> bool:
        """
        Check if two specific component versions are compatible.

        Returns false until proven compatible - security through pessimism.
        """
        # Check if we have direct compatibility data
        if (component1 in self._compatibility_map and
            component2 in self._compatibility_map[component1]):
            for c1_ver, c2_ver in self._compatibility_map[component1][component2]:
                # Check for exact version match or compatibility based on semver
                if c1_ver == version1 and c2_ver == version2:
                    return True

                # For exact match of first component, check if second is compatible
                if c1_ver == version1:
                    try:
                        if is_compatible(version2, c2_ver):
                            return True
                    except Exception as e:
                        logger.debug(f"Compatibility check failed: {e}")

        # If no direct information, assume incompatible - fail closed
        return False

    def get_compatible_versions(self, component: str, version: str) -> Dict[str, List[str]]:
        """Get all components/versions compatible with the specified component version."""
        result: Dict[str, List[str]] = {}

        if component not in self._compatibility_map:
            return result  # Return empty dictionary for unknown components

        for target_comp, compat_versions in self._compatibility_map[component].items():
            compatible_versions: List[str] = []

            for comp_ver, target_ver in compat_versions:
                if comp_ver == version:
                    compatible_versions.append(target_ver)

            if compatible_versions:
                result[target_comp] = compatible_versions

        return result

    def generate_compatibility_report(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """
        Generate a complete compatibility report for documentation.

        Returns a nested dictionary structure mapping components to their
        compatible targets, organized by version.
        """
        report: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

        for component, targets in self._compatibility_map.items():
            report[component] = {}

            for target_comp, compat_versions in targets.items():
                # Group by source version
                by_version: Dict[str, List[str]] = {}

                for comp_ver, target_ver in compat_versions:
                    if comp_ver not in by_version:
                        by_version[comp_ver] = []
                    by_version[comp_ver].append(target_ver)

                report[component][target_comp] = by_version

        return report

    def _create_graphical_visualization(self, output_path: Optional[str] = None) -> str:
        """
        Create graphical visualization with matplotlib if available.

        Gracefully handles optional dependency availability.
        """
        try:
            # Dynamic imports to handle optional dependencies
            import matplotlib.pyplot as plt_dynamic  # Use different name to avoid conflict
            import numpy as np

            # Type hint for the dynamic import to make Pylance happy
            plt = cast(PyplotProto, plt_dynamic)  # Explicit cast clarifies intent

            # Get all unique components
            components: List[str] = sorted(self._compatibility_map.keys())
            n_components: int = len(components)

            if n_components == 0:
                return "No components registered for visualization. Matrix is empty."

            # Create compatibility matrix
            matrix: FloatingArray = np.zeros((n_components, n_components))

            # Fill matrix with compatibility information
            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components):
                    if comp1 == comp2:
                        matrix[i, j] = 1  # Self-compatible by definition
                    elif (comp1 in self._compatibility_map and
                          comp2 in self._compatibility_map[comp1]):
                        matrix[i, j] = len(self._compatibility_map[comp1][comp2])

            # Create visualization with elegant styling
            _ = plt.figure(figsize=(10, 8))  # Use _ for unused variable
            img = plt.matshow(matrix, cmap='Blues')
            _ = plt.colorbar(mappable=img, label='Number of compatible version pairs')

            # Set component labels with perfect alignment
            components_seq: Sequence[str] = cast(Sequence[str], components)
            plt.xticks(range(n_components), components_seq, rotation=90)
            plt.yticks(range(n_components), components_seq)

            # Add descriptive title
            plt.title('Eidosian Component Compatibility Matrix')
            plt.tight_layout()

            # Save if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                return f"Visualization saved to {output_path} with elegance intact"
            else:
                # Display in interactive mode
                plt.show()
                return "Compatibility visualization displayed in all its glory"

        except ImportError as e:
            return f"Could not create graphical visualization: {e}. Install matplotlib and numpy for visual elegance."
        except Exception as e:
            return f"Error creating visualization: {e}"

    def _create_ascii_visualization(self) -> str:
        """
        Create an ASCII art visualization of the compatibility matrix.

        When pixels fail, characters prevail - the eternal fallback.
        """
        components = sorted(self._compatibility_map.keys())

        if not components:
            return "No components registered in compatibility matrix. The void stares back."

        # Calculate column width based on longest component name
        col_width = max(len(comp) for comp in components) + 2

        # Build header row with perfect alignment
        header = ' ' * col_width + '│ ' + ' '.join(comp.ljust(col_width) for comp in components)
        separator = '─' * col_width + '┼' + '─' * (len(header) - col_width - 1)

        # Build rows with relationship indicators
        rows: List[str] = []
        for comp1 in components:
            row: List[str] = [comp1.ljust(col_width) + '│']

            for comp2 in components:
                if comp1 == comp2:
                    cell = '✓'.center(col_width)  # Self-compatible
                elif (comp1 in self._compatibility_map and
                      comp2 in self._compatibility_map[comp1]):
                    count = len(self._compatibility_map[comp1][comp2])
                    cell = (f"{count}").center(col_width)
                else:
                    cell = '·'.center(col_width)  # Empty but visually meaningful

                row.append(cell)

            rows.append(' '.join(row))

        # Combine all parts into a structural masterpiece
        ascii_viz = '\n'.join([header, separator] + rows)
        return ascii_viz

    def to_json(self) -> str:
        """
        Export compatibility matrix as JSON for perfect serialization.

        Structural preservation in transit - data as immutable truth.
        """
        return json.dumps({
            "components": self._versions,
            "compatibility": self._compatibility_map
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CompatibilityMatrix':
        """
        Import compatibility matrix from JSON with perfect reconstruction.

        Factory method for hydrating matrix from serialized form.
        """
        matrix = cls()
        try:
            data = json.loads(json_str)

            # Restore version information
            matrix._versions = data.get("components", {})

            # Restore compatibility mappings
            matrix._compatibility_map = data.get("compatibility", {})

            logger.debug(f"Restored compatibility matrix with {len(matrix._versions)} components")
            return matrix
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse compatibility matrix JSON: {e}")
            return matrix  # Return empty matrix rather than failing
        except Exception as e:
            logger.error(f"Unexpected error restoring compatibility matrix: {e}")
            return matrix  # Return empty matrix as fallback
