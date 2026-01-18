"""
Version Forge - Semantic versioning and compatibility enforcement.
Ensures Eidosian components align with structural evolution.
"""
import re
from typing import Dict, Any, List, Optional, Tuple, Union

class Version:
    """
    Representation of a Semantic Version (SemVer).
    Format: major.minor.patch[-prerelease][+build]
    """
    
    # SemVer 2.0.0 Regex
    SEMVER_PATTERN = re.compile(
        r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
        r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )

    def __init__(self, version_str: str):
        match = self.SEMVER_PATTERN.match(version_str)
        if not match:
            raise ValueError(f"Invalid SemVer string: {version_str}")
        
        self.major = int(match.group("major"))
        self.minor = int(match.group("minor"))
        self.patch = int(match.group("patch"))
        self.prerelease = match.group("prerelease")
        self.build = match.group("buildmetadata")
        self.version_str = version_str

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Version):
            return False
        return (self.major, self.minor, self.patch, self.prerelease) == \
               (other.major, other.minor, other.patch, other.prerelease)

    def __lt__(self, other: 'Version') -> bool:
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        
        # Prerelease handling (simplified: presence < absence)
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
            
        return False

    def __le__(self, other: 'Version') -> bool:
        return self < other or self == other

    def __gt__(self, other: 'Version') -> bool:
        return not self <= other

    def __ge__(self, other: 'Version') -> bool:
        return not self < other

    def __repr__(self) -> str:
        return f"Version({self.version_str})"

class VersionForge:
    """
    Manages component versions and compatibility matrices.
    """
    
    def __init__(self):
        self._components: Dict[str, Version] = {}
        # compatibility_map: {component: {dependency: required_version_spec}}
        self._compatibility_map: Dict[str, Dict[str, str]] = {}

    def register_component(self, name: str, version: Union[str, Version]):
        """Register a component with its current version."""
        if isinstance(version, str):
            version = Version(version)
        self._components[name] = version

    def get_version(self, name: str) -> Optional[Version]:
        """Get the version of a registered component."""
        return self._components.get(name)

    def check_compatibility(self, name: str, dependency: str, required_spec: str) -> bool:
        """
        Check if a component's dependency meets the required specification.
        Supported specs: ^major.minor.patch (Compatible), ~major.minor.patch (Patch only), == (Exact)
        """
        dep_version = self.get_version(dependency)
        if not dep_version:
            return False
        
        if required_spec.startswith('^'):
            req = Version(required_spec[1:])
            return dep_version.major == req.major and dep_version >= req
        elif required_spec.startswith('~'):
            req = Version(required_spec[1:])
            return dep_version.major == req.major and dep_version.minor == req.minor and dep_version >= req
        elif required_spec.startswith('=='):
            req = Version(required_spec[2:])
            return dep_version == req
        else:
            # Default to exact if no prefix
            req = Version(required_spec)
            return dep_version == req

    def validate_system(self) -> List[str]:
        """Validate all registered dependencies. Returns list of errors."""
        errors = []
        for component, deps in self._compatibility_map.items():
            for dep, spec in deps.items():
                if not self.check_compatibility(component, dep, spec):
                    errors.append(f"Incompatibility: {component} requires {dep} {spec}, but found {self.get_version(dep)}")
        return errors

    def add_dependency(self, component: str, dependency: str, spec: str):
        """Add a dependency requirement to the compatibility map."""
        if component not in self._compatibility_map:
            self._compatibility_map[component] = {}
        self._compatibility_map[component][dependency] = spec
