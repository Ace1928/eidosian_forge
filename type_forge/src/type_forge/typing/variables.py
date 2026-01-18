"""
Type Variables for the Type Forge system.

"""

from __future__ import annotations

from typing import Any  # Used only for Callable type parameters where truly generic
from typing import Callable, Hashable, TypeVar

from type_forge.typing.protocols import SupportsComparison

from . import __version__  # noqa: F401

version = __version__

# ──────────────────────────────────────────────────────────────
# Core Type Variables
# ──────────────────────────────────────────────────────────────

# Generic type variables with precise constraints
T = TypeVar("T")  # Generic invariant type variable
T_co = TypeVar("T_co", covariant=True)  # Covariant type (produces values)
T_contra = TypeVar(
    "T_contra",
    contravariant=True,
)  # Contravariant type (consumes values)

# Source/Result type variables for transformations
S = TypeVar("S")  # Source type invariant
S_co = TypeVar("S_co", covariant=True)  # Source covariant type
S_contra = TypeVar("S_contra", contravariant=True)  # Source contravariant type

# Result type variables
R = TypeVar("R")  # Result type invariant
R_co = TypeVar("R_co", covariant=True)  # Result covariant type
R_contra = TypeVar("R_contra", contravariant=True)  # Result contravariant type

# Utility type variables
U = TypeVar("U")  # Utility invariant type
U_co = TypeVar("U_co", covariant=True)  # Utility covariant type
U_contra = TypeVar("U_contra", contravariant=True)  # Utility contravariant type

# Key-Value type variables for mappings
K = TypeVar("K", bound=Hashable)  # Key type invariant (must be hashable)
K_co = TypeVar("K_co", bound=Hashable, covariant=True)  # Key covariant type
K_contra = TypeVar(
    "K_contra",
    bound=Hashable,
    contravariant=True,
)  # Key contravariant type

V = TypeVar("V")  # Value type invariant
V_co = TypeVar("V_co", covariant=True)  # Value covariant type
V_contra = TypeVar("V_contra", contravariant=True)  # Value contravariant type

# Specialized type variables with bounds
HashableT = TypeVar("HashableT", bound=Hashable)  # Type constrained to be Hashable
HashableT_co = TypeVar(
    "HashableT_co",
    bound=Hashable,
    covariant=True,
)  # Covariant hashable type
HashableT_contra = TypeVar(
    "HashableT_contra",
    bound=Hashable,
    contravariant=True,
)  # Contravariant hashable type

ComparableT = TypeVar(
    "ComparableT",
    bound=SupportsComparison,
)  # Type must support comparison
ComparableT_co = TypeVar(
    "ComparableT_co",
    bound=SupportsComparison,
    covariant=True,
)  # Covariant comparable
ComparableT_contra = TypeVar(
    "ComparableT_contra",
    bound=SupportsComparison,
    contravariant=True,
)  # Contravariant comparable

# Instance and value manipulation type variables
TInstance = TypeVar("TInstance")  # Instance type for creation methods
TInstance_co = TypeVar("TInstance_co", covariant=True)  # Covariant instance type
TInstance_contra = TypeVar(
    "TInstance_contra",
    contravariant=True,
)  # Contravariant instance type

TValue = TypeVar("TValue")  # Value type for parameters
TValue_co = TypeVar("TValue_co", covariant=True)  # Covariant value type
TValue_contra = TypeVar("TValue_contra", contravariant=True)  # Contravariant value type

# Collection specific type variables
TCollection = TypeVar("TCollection")  # Collection type
TCollection_co = TypeVar("TCollection_co", covariant=True)  # Covariant collection type
TCollection_contra = TypeVar(
    "TCollection_contra",
    contravariant=True,
)  # Contravariant collection type

# Function specific type variables - fixing missing Callable type params
TCallable = TypeVar("TCallable", bound=Callable[..., Any])  # Callable type
TCallable_co = TypeVar(
    "TCallable_co",
    bound=Callable[..., Any],
    covariant=True,
)  # Covariant callable type
TCallable_contra = TypeVar(
    "TCallable_contra",
    bound=Callable[..., Any],
    contravariant=True,
)  # Contravariant callable type

# Error and exception handling type variables
TError = TypeVar("TError", bound=Exception)  # Error type
TError_co = TypeVar(
    "TError_co",
    bound=Exception,
    covariant=True,
)  # Covariant error type
TError_contra = TypeVar(
    "TError_contra",
    bound=Exception,
    contravariant=True,
)  # Contravariant error type
