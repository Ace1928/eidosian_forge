# filepath: /home/lloyd/repos/global_info.py
import os
import platform
import pprint
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict, TypeVar, Union

"""
╔═══════════════════════════════════════════════════════════════════════════╗
║   ███████╗  ██╗  ██████╗    ██████╗   ███████╗  ██╗   █████╗   ███╗   ██╗ ║
║   ██╔════╝  ██║  ██╔══██╗  ██╔═══██╗  ██╔════╝  ██║  ██╔══██╗  ████╗  ██║ ║
║   █████╗    ██║  ██║  ██║  ██║   ██║  ███████╗  ██║  ███████║  ██╔██╗ ██║ ║
║   ██╔══╝    ██║  ██║  ██║  ██║   ██║  ╚════██║  ██║  ██╔══██║  ██║╚██╗██║ ║
║   ███████╗  ██║  ██████╔╝  ╚██████╔╝  ███████║  ██║  ██║  ██║  ██║ ╚████║ ║
║   ╚══════╝  ╚═╝  ╚═════╝    ╚═════╝   ╚══════╝  ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═══╝ ║
║                                                                           ║
║ ╭────────────────────────────────────────────────────────────────────╮    ║
║ │           G L O B A L   I N F O R M A T I O N   S Y S T E M        │    ║
║ ╰────────────────────────────────────────────────────────────────────╯    ║
╟───────────────────────────────────────────────────────────────────────────╢
║  Centralized configuration for all Eidosian repos and projects.           ║
║  Maintainer: Lloyd Handyside (ace1928@gmail.com)                          ║
║  Organization: Neuroforge (lloyd.handyside@neuroforge.io)                 ║
║  Contributor: Eidos (syntheticeidos@gmail.com)                            ║
║  GitHub: github.com/Ace1928                                               ║
║  License: MIT (OSI compliant, see LICENSE text below)                     ║
║  Python: 3.12                                                             ║
║  Created: 12/3/2025                                                       ║
║  Last Modified: ${date.today().strftime('%Y-%m-%d')}                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 🖥️ SYSTEM INFORMATION                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
System information about the underlying platform and environment.
Knowledge begins with precise awareness of computational context. 🧠
"""
SYSTEM_INFO: Dict[str, str] = {
    "platform": platform.system(),
    "platform_release": platform.release(),
    "platform_version": platform.version(),
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "hostname": platform.node(),
    "boot_time": "N/A",  # Populated at runtime if needed
    "os_distribution": "N/A",  # Populated at runtime if needed
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 👤 AUTHOR & ORGANIZATION INFORMATION                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Author and organization details for identification and contact purposes.
Digital creations require lineage—code doesn't spontaneously manifest. 🧬
"""
AUTHOR_INFO: Dict[str, str] = {
    "name": "Lloyd Handyside",
    "email": "ace1928@gmail.com",
    "org": "Neuroforge",
    "org_email": "lloyd.handyside@neuroforge.io",
    "org_website": "neuroforge.io",
    "contributor": "Eidos",
    "contributor_email": "syntheticeidos@gmail.com",
    "github_user": "Ace1928",
    "github_org": "",
    "gitlab_user": "ace1928",
    "twitter": "@NeuroforgeAI",
    "linkedin": "lloyd-handyside",
    "orcid": "0000-0000-0000-0000",  # Placeholder—identity without verification is merely assertion
    "pgp_key": "XXXX-XXXX-XXXX-XXXX",  # Placeholder—security theater until populated
    "timezone": "Europe/London",
    "locale": "en_GB",
    "personal_motto": "Perfection is not an aspiration—it is the default.",
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 📜 LICENSING INFORMATION                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Full text of the MIT license for distribution compliance.
Freedom requires precise boundaries—permission without ambiguity. ⚖️
"""
MIT_LICENSE: str = f"""MIT License

Copyright (c) {date.today().year} {AUTHOR_INFO["name"]} / {AUTHOR_INFO["org"]}

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# License placeholders—verbose when needed, concise when not
APACHE_LICENSE: str = "Apache License 2.0"  # Expand when implementation requires it
GPL_LICENSE: str = "GNU General Public License v3.0"  # Copyleft, but only when relevant
BSD_LICENSE: str = "BSD 3-Clause License"  # Simplicity has its place
CC_LICENSE: str = "Creative Commons Attribution 4.0"  # For content, not code

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 🗂️ FILESYSTEM ARCHITECTURE & ENVIRONMENT TOPOLOGY                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Digital cartography—precise path definitions for confident navigation.
Every byte needs an address; every function needs a home. 🧭
"""
# Core filesystem topology—the skeletal structure of our digital organism
ROOT_DIR: Path = Path(__file__).parent.resolve()  # Truth begins here
VENV_DIR: Path = ROOT_DIR / "eidos_venv"  # Isolation chamber for dependencies
DATA_DIR: Path = ROOT_DIR / "data"  # Where information goes to become knowledge
CACHE_DIR: Path = ROOT_DIR / ".cache"  # Temporary insights, persistent patterns
LOGS_DIR: Path = ROOT_DIR / "logs"  # Digital breadcrumbs for future archeologists
CONFIG_DIR: Path = ROOT_DIR / "config"  # The rules that govern reality
TEMP_DIR: Path = ROOT_DIR / "tmp"  # Ephemeral staging ground
ASSETS_DIR: Path = ROOT_DIR / "assets"  # Static treasures, immutable artifacts
MODELS_DIR: Path = ROOT_DIR / "models"  # Intelligence, crystallized
DOCS_BUILD_DIR: Path = ROOT_DIR / "docs" / "_build"  # Knowledge in presentable form
BACKUP_DIR: Path = ROOT_DIR / "backup"  # Insurance against entropy

# Version constraints—minimal requirements for technological participation
PYTHON_VERSION: str = ">=3.12"  # Only the future-proof need apply
NODE_VERSION: str = ">=18.0.0"  # JavaScript with fewer historical wounds
GO_VERSION: str = ">=1.20.0"  # Speed without sacrificing readability
RUST_VERSION: str = ">=1.70.0"  # Safety without sacrificing performance

"""
Environment differentiators—context-aware configuration patterns.
Same code, different behavior—quantum superposition for deployment states. ⚛️
"""
ENV_CONFIGS: Dict[str, Dict[str, str | bool]] = {
    "development": {
        "debug": True,  # Verbose clarity trumps elegant silence
        "log_level": "DEBUG",  # Maximum visibility in familiar territory
        "profile": "dev",  # The sandbox where consequences are temporary
    },
    "testing": {
        "debug": True,  # Trust but verify
        "log_level": "INFO",  # Signal above noise
        "profile": "test",  # The proving ground where assumptions meet reality
    },
    "staging": {
        "debug": False,  # Dress rehearsal with production costumes
        "log_level": "INFO",  # Important events still matter
        "profile": "stage",  # The final checkpoint before the spotlight
    },
    "production": {
        "debug": False,  # In production, silence is golden
        "log_level": "WARNING",  # Only speak when something's wrong
        "profile": "prod",  # Where theory meets consequence
    },
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 📦 DEPENDENCIES & TOOLKIT REQUIREMENTS                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Package ecosystem topology—precise dependency vectors for deterministic builds.
Digital supply chain mapped with version constraints for reproducible reality. 🧪
"""

# ┌────────────────────────────────────────────────────────┐
# │ Development Arsenal — Weapons for Code Crafting        │
# └────────────────────────────────────────────────────────┘
DEV_DEPENDENCIES: List[str] = [
    "black",  # Code formatter that brooks no argument
    "isort",  # Import wrangler for alphabetical sanity
    "flake8",  # Linter with opinions (usually correct)
    "mypy",  # Type checking for the paranoid perfectionist
    "pytest",  # Where code meets its maker
    "pytest-cov",  # Trust through verification, not optimism
    "build",  # Construction without the dust
    "twine",  # Package delivery service
    "pre-commit",  # Gatekeeping for git transgressions
    "bumpversion",  # Incrementing with purpose
    "tox",  # Multi-environment torture chamber
    "bandit",  # Security scanner for the justifiably suspicious
    "safety",  # Vulnerability detector for the cautiously competent
    "detect-secrets",  # Because secrets belong in vaults, not repositories
    "pyright",  # Microsoft's opinion on your types
    "debugpy",  # Surgery tools for sick code
]

# ┌────────────────────────────────────────────────────────┐
# │ Documentation Apparatus — Knowledge Crystallization    │
# └────────────────────────────────────────────────────────┘
DOCS_DEPENDENCIES: List[str] = [
    "sphinx>=8.2.3",  # Documentation that assumes intelligence
    "furo>=2024.8.6",  # Theme that respects cognitive bandwidth
    "sphinx-rtd-theme>=3.0.2",  # For when tradition matters
    "myst-parser>=4.0.1",  # Markdown support for the modern mind
    "sphinxcontrib-plantuml>=0.2.0",  # Diagrams for the visually articulate
    "sphinxcontrib-mermaid>=0.1.0",  # When flowcharts beat paragraphs
    "sphinx-autoapi>=3.0.0",  # Documentation that writes itself
    "sphinx-copybutton>=0.5.2",  # Clipboard integration for the impatient
    "sphinx-autodoc-typehints>=3.1.0",  # Types as documentation, not decoration
    "sphinx-autobuild>=2024.10.3",  # Live preview for the immediate gratification
    "sphinx-design>=0.6.1",  # Because aesthetics and function coexist
    "mkdocs>=1.5.0",  # Alternative documentation for the undecided
    "mkdocs-material>=9.5.0",  # Material design for material minds
    "mkdocstrings>=0.24.0",  # String extraction without the tedium
    "pdoc>=14.3.0",  # Documentation with minimal intervention
]

# ┌────────────────────────────────────────────────────────┐
# │ Core Capabilities — Foundational Digital Substrate     │
# └────────────────────────────────────────────────────────┘
CORE_DEPENDENCIES: List[str] = [
    "requests>=2.31.0",  # HTTP for humans, not machines
    "aiohttp>=3.11.0",  # Asynchronicity for the impatient
    "colorama>=0.4.6",  # Terminal aesthetics for the visually demanding
    "numpy>=1.26.0",  # Mathematics at electronic speed
    "packaging>=24.0",  # Meta-packaging for package packagers
    "pydantic>=2.6.0",  # Types with enforcement capabilities
    "python-dotenv>=1.0.0",  # Environment variables without environment variables
    "typer>=0.9.0",  # CLIs that don't feel like punishment
    "rich>=13.7.0",  # Terminal output for the 21st century
    "click>=8.1.7",  # Commands with ergonomics
    "tqdm>=4.66.2",  # Progress bars that respect your attention
    "pyinstrument>=4.6.1",  # Profiling without a PhD requirement
]

# ┌────────────────────────────────────────────────────────┐
# │ Data Processing — Information Transmutation Tools      │
# └────────────────────────────────────────────────────────┘
DATA_DEPENDENCIES: List[str] = [
    "pandas>=2.2.0",  # Tabular data for the spreadsheet-averse
    "scipy>=1.12.0",  # Science minus the lab coat
    "matplotlib>=3.8.0",  # Plotting for when words fail
    "seaborn>=0.13.0",  # Statistical visualization with taste
    "scikit-learn>=1.4.0",  # Machine learning without the PhD
    "statsmodels>=0.14.1",  # Statistics when you need to prove a point
]

# ┌────────────────────────────────────────────────────────┐
# │ Machine Learning — Artificial Intelligence Substrate   │
# └────────────────────────────────────────────────────────┘
ML_DEPENDENCIES: List[str] = [
    "torch>=2.2.0",  # PyTorch: tensors all the way down
    "tensorflow>=2.15.0",  # Google's gradient calculator
    "jax>=0.4.20",  # Derivatives with a side of XLA
    "transformers>=4.37.0",  # NLP without the linguistics degree
    "huggingface-hub>=0.20.0",  # Model sharing for the generous
    "datasets>=2.16.0",  # Data management for the overwhelmed
    "accelerate>=0.27.0",  # Speed without the optimization headaches
]

# ┌────────────────────────────────────────────────────────┐
# │ Web Infrastructure — Digital Presence Toolkit          │
# └────────────────────────────────────────────────────────┘
WEB_DEPENDENCIES: List[str] = [
    "fastapi>=0.110.0",  # APIs with velocity and validation
    "uvicorn>=0.27.0",  # ASGI server that means business
    "flask>=3.0.0",  # Microframework for the minimalist
    "django>=5.0.0",  # Framework for the kitchen sink enthusiast
    "starlette>=0.36.0",  # ASGI framework for the performance-conscious
    "httpx>=0.26.0",  # HTTP client from the future
]

# ┌────────────────────────────────────────────────────────┐
# │ Cloud Integration — Provider-Specific Connectors       │
# └────────────────────────────────────────────────────────┘
CLOUD_DEPENDENCIES: Dict[str, List[str]] = {
    "aws": [
        "boto3>=1.34.0",  # Amazon's cloud API wrapped for Python
        "awscli>=1.32.0",  # Command line for when GUIs fail you
    ],
    "azure": [
        "azure-storage-blob>=12.19.0",  # Microsoft's object storage interface
        "azure-identity>=1.15.0",  # Authentication without the ceremony
    ],
    "gcp": [
        "google-cloud-storage>=2.14.0",  # Google's bytes warehouse
        "google-auth>=2.27.0",  # Permission slips for Google's cloud
    ],
}

# ┌────────────────────────────────────────────────────────┐
# │ Dependency Matrix — Full Requirements Topology         │
# └────────────────────────────────────────────────────────┘
FULL_DEPENDENCY_MAP: Dict[str, List[str]] = {
    "core": CORE_DEPENDENCIES,
    "dev": DEV_DEPENDENCIES,
    "docs": DOCS_DEPENDENCIES,
    "data": DATA_DEPENDENCIES,
    "ml": ML_DEPENDENCIES,
    "web": WEB_DEPENDENCIES,
    **{f"cloud.{k}": v for k, v in CLOUD_DEPENDENCIES.items()},
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 🐳 CONTAINERIZATION & ORCHESTRATION                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Deployment topology for hermetically sealed code execution environments.
Ship your code like a well-packaged virus—contained, complete, and ready to replicate. 🚢
"""

# Immutable topology for reproducible execution environments
CONTAINERIZATION: Dict[str, Dict[str, Any]] = {
    "docker": {
        "base_images": {
            "python": "python:3.12-slim",  # Minimal surface area for attacks
            "node": "node:18-alpine",  # JavaScript with a mountain view
            "go": "golang:1.20-alpine",  # Performance in a puffer jacket
            "rust": "rust:1.70-slim",  # Safety without the comfort padding
        },
        "registry": "ghcr.io/neuroforge",  # Version-control-adjacent artifacts
        "labels": {  # Metadata—because unlabeled containers are digital mysteries
            "org.opencontainers.image.source": "https://github.com/Neuroforge/",
            "org.opencontainers.image.vendor": AUTHOR_INFO["org"],
            "org.opencontainers.image.licenses": "MIT",
        },
        "build_args": {  # Build-time variables—configuration injection at genesis
            "PYTHON_VERSION": "3.12",
            "NODE_ENV": "production",
            "BUILD_DATE": "${BUILD_DATE}",  # Template for CI injection
        },
    },
    "docker_compose": {  # Multi-container development topology
        "version": "3.9",  # API version, not aesthetic preference
        "services": {},  # Populated at project instantiation
        "networks": {  # Isolation boundaries for service communication
            "frontend": {"driver": "bridge"},
            "backend": {"driver": "bridge", "internal": True},
        },
    },
    "kubernetes": {  # Production orchestration substrate
        "namespace": "eidosian",  # Logical partition in the cluster multiverse
        "common_labels": {  # Taxonomic markers for resource identification
            "app.kubernetes.io/part-of": "eidosian",
            "app.kubernetes.io/managed-by": "neuroforge",
        },
        "ingress": {  # Border control for digital immigrants
            "controller": "nginx",
            "tls": "enabled",  # Encryption isn't optional, it's architectural
            "annotations": {  # Whispers to the orchestrator
                "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                "nginx.ingress.kubernetes.io/ssl-redirect": "true",
            },
        },
        "resources": {  # Resource allocation contracts—promises to the scheduler
            "requests": {  # Minimum viable resources—what your app needs to breathe
                "cpu": "100m",
                "memory": "128Mi",
            },
            "limits": {  # Resource ceiling—preventing noisy neighbors
                "cpu": "500m",
                "memory": "512Mi",
            },
        },
    },
}

# Deployment domain registry—where containers materialize across environments
DEPLOYMENT_TARGETS: Dict[str, Dict[str, str]] = {
    "development": {  # Where mistakes are cheap and learning is required
        "kubernetes_context": "minikube",
        "registry": CONTAINERIZATION["docker"]["registry"],
        "domain_suffix": ".dev.local",
    },
    "staging": {  # The dress rehearsal before the production premiere
        "kubernetes_context": "staging-cluster",
        "registry": CONTAINERIZATION["docker"]["registry"],
        "domain_suffix": ".staging.neuroforge.io",
    },
    "production": {  # Where theory meets consequence—and customers
        "kubernetes_context": "production-cluster",
        "registry": CONTAINERIZATION["docker"]["registry"],
        "domain_suffix": ".neuroforge.io",
    },
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 💅 CODE STYLE & LINTING                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Style guidelines for code that reads like it wrote itself.
When aesthetics and functionality converge, bugs have nowhere to hide. 📝
"""
CODE_STYLE: Dict[str, Dict[str, Any]] = {
    "python": {
        "line_length": 88,  # Long enough for expression, short enough for comprehension
        "black": {
            "skip_string_normalization": True,  # Single quotes deserve freedom too
            "target_version": ["py312"],  # The future is now
            "exclude": r"/(\.git|\.venv|venv|node_modules|build|dist)/",  # Digital wastelands
        },
        "isort": {
            "profile": "black",  # Consistency is a virtue, not a choice
            "multi_line_output": 3,  # The Zen of import organization
            "include_trailing_comma": True,  # Prepare for the future, one comma at a time
            "force_grid_wrap": 0,  # Let imports breathe naturally
            "use_parentheses": True,  # Embrace your imports like they're family
            "ensure_newline_before_comments": True,  # Comments deserve personal space
        },
        "flake8": {
            "ignore": ["E203", "E501", "W503"],  # Some rules exist to be broken
            "max-line-length": 88,  # The universe and black agree on this number
            "exclude": [
                ".git",
                "__pycache__",
                "build",
                "dist",
                ".venv",
                "venv",
            ],  # The untouchables
        },
        "mypy": {
            "python_version": "3.12",  # Type-checking from the future
            "warn_return_any": True,  # 'Any' is not a type, it's a surrender
            "warn_unused_configs": True,  # Dead configs tell no tales
            "disallow_untyped_defs": True,  # Types or it didn't happen
            "disallow_incomplete_defs": True,  # Partial typing is like partial truth
            "check_untyped_defs": True,  # Trust no function, verify all types
        },
    },
    "javascript": {
        "eslint": {
            "extends": [
                "eslint:recommended",
                "plugin:@typescript-eslint/recommended",
            ],  # Standing on giants' shoulders
            "parser": "@typescript-eslint/parser",  # The interpreter of interpreted languages
        },
        "prettier": {
            "printWidth": 100,  # JavaScript deserves more breathing room
            "tabWidth": 2,  # Indentation: minimal but meaningful
            "semi": True,  # Statements end with certainty, not ambiguity
            "singleQuote": True,  # Double the quote, double the keystrokes
            "trailingComma": "es5",  # The comma that prevents git blame wars
        },
    },
    "go": {
        "gofmt": True,  # Go doesn't ask, it formats
        "golint": True,  # Linting so native it's practically built-in
    },
    "rust": {
        "clippy": True,  # Your code's best friend and harshest critic
        "rustfmt": {
            "edition": "2021",  # Modern times require modern formatting
            "max_width": 100,  # Width to match our JavaScript friends
        },
    },
    "markdown": {
        "line_length": 120,  # Prose deserves more space than code
        "emphasis_style": "*",  # Asterisks hit the sweet spot of visibility
        "list_indent": 2,  # The perfect hierarchical increment
    },
    "general": {
        "tabs_vs_spaces": "spaces",  # Tabs are width-inconsistent space vampires
        "indent_size": 4,  # Python's democracy of whitespace
        "trim_trailing_whitespace": True,  # Digital hygiene 101
        "end_of_line": "lf",  # CRLF is redundancy incarnate
        "charset": "utf-8",  # Because ASCII is culturally insensitive
    },
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 🧪 TESTING SPECIFICATIONS & VERIFICATION PROTOCOLS                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Test execution parameters with precise verification boundaries.
Without tests, your code is just a collection of hopeful assumptions. 🔬
"""


# ┌────────────────────────────────────────────────────────┐
# │ Type Definitions — Contracts for Testing Configuration │
# └────────────────────────────────────────────────────────┘
class PytestMarkerConfig(TypedDict):
    """Configuration for pytest markers and execution options."""

    markers: List[str]  # Test taxonomies for selective execution
    options: Dict[str, Union[List[str], str, bool]]  # Runtime parameters


class CoverageConfig(TypedDict):
    """Configuration for code coverage requirements."""

    required_percentage: int  # Minimum acceptable test coverage
    excludes: List[str]  # Paths granted temporary immunity
    branch: bool  # Whether to measure decision path coverage


class PathConfig(TypedDict):
    """Configuration for test resource locations."""

    data_dir: str  # Path to test data
    snapshot_dir: str  # Path to expected outputs


class MockConfig(TypedDict):
    """Configuration for test doubles."""

    services_dir: str  # Internal service simulacra
    external_apis_dir: str  # External dependency mimics


class StrategyConfig(TypedDict):
    """Configuration for test execution strategies."""

    default: str  # Standard testing posture
    ci: str  # Continuous integration approach
    local: str  # Development environment approach


class TestingConfig(TypedDict):
    """Complete testing configuration topology."""

    pytest: PytestMarkerConfig
    coverage: CoverageConfig
    fixtures: PathConfig
    mocks: MockConfig
    strategies: StrategyConfig


# ┌────────────────────────────────────────────────────────┐
# │ Testing Configuration — The Rules of Engagement        │
# └────────────────────────────────────────────────────────┘
TESTING_CONFIGS: TestingConfig = {
    "pytest": {
        "markers": [
            "unit: mark a test as a unit test",  # Quantum-level reality checks
            "integration: mark a test as an integration test",  # Component interplay validation
            "e2e: mark a test as an end-to-end test",  # Full system simulation
            "slow: mark a test as slow running",  # Tests that outlive fruit flies
            "property: mark a test that verifies invariants",  # Mathematical certainty proofs
            "security: mark tests that validate defenses",  # Digital immune system trials
        ],
        "options": {
            "testpaths": ["tests"],  # Where assumptions meet verification
            "python_files": "test_*.py",  # The naming ritual for test discovery
            "python_classes": "Test*",  # Classes that judge your code's sins
            "python_functions": "test_*",  # Functions that trust nothing, verify everything
            "xvs": True,  # Extra verbosity for extra wisdom
        },
    },
    "coverage": {
        "required_percentage": 85,  # The threshold between confidence and delusion
        "excludes": [
            "tests/*",  # Testing test code: an infinite mirror trap
            "setup.py",  # Infrastructure code tests itself by existing
            "conftest.py",  # Test fixtures prove themselves by serving
            "**/deprecated/**",  # Code awaiting digital cremation
        ],
        "branch": True,  # Every decision fork must face scrutiny
    },
    "fixtures": {
        "data_dir": str(
            ROOT_DIR / "tests" / "fixtures" / "data"
        ),  # Input simulation chamber
        "snapshot_dir": str(
            ROOT_DIR / "tests" / "fixtures" / "snapshots"
        ),  # Expected reality blueprints
    },
    "mocks": {
        "services_dir": str(
            ROOT_DIR / "tests" / "mocks" / "services"
        ),  # Internal service body doubles
        "external_apis_dir": str(
            ROOT_DIR / "tests" / "mocks" / "external"
        ),  # Third-party dependency clones
    },
    "strategies": {
        "default": "thorough",  # Trust, but verify exhaustively
        "ci": "paranoid",  # In automation we trust nothing
        "local": "pragmatic",  # Fast enough to run, thorough enough to matter
    },
}

# ┌────────────────────────────────────────────────────────┐
# │ Test Execution Matrix — The Speed vs. Truth Tradeoff   │
# └────────────────────────────────────────────────────────┘
TEST_EXECUTION_MATRIX: Dict[str, Dict[str, Union[int, Literal["all"]]]] = {
    "quick": {
        "unit": "all",  # Unit tests: the non-negotiable foundation
        "integration": 0,  # Integration tests: sacrificed for velocity
        "e2e": 0,  # End-to-end tests: too slow for this mode
    },
    "normal": {
        "unit": "all",  # All cells must function
        "integration": "all",  # All organs must cooperate
        "e2e": 10,  # Sample the full organism's behavior
    },
    "thorough": {
        "unit": "all",  # Every function must prove itself
        "integration": "all",  # Every connection must demonstrate integrity
        "e2e": "all",  # Every workflow must survive simulation
    },
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 🚀 CI/CD CONFIGURATIONS                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Continuous Integration/Delivery settings for automated validation and deployment.
Human error is inevitable; automation is how we contain it. ⚙️
"""

# ┌────────────────────────────────────────────────────────┐
# │ CI/CD Settings — Machines Judging Machine-Made Code    │
# └────────────────────────────────────────────────────────┘
CICD_CONFIGS: Dict[str, Dict[str, Any]] = {
    "github_actions": {
        "python_versions": ["3.12"],  # Only the future-proof need apply
        "os": [
            "ubuntu-latest",
            "macos-latest",
            "windows-latest",
        ],  # The trinity of OS truth
        "branches": ["main", "develop"],  # Branches worthy of automation attention
    },
    "gitlab_ci": {
        "default_image": "python:3.12-alpine",  # Minimalism in a container
        "stages": ["test", "build", "deploy"],  # The triumvirate of delivery
    },
    "jenkins": {
        "agent": "docker",  # Containers containing continuous integration
        "jdk": "openjdk17",  # Java that doesn't feel like a time capsule
    },
    "release": {
        "branches": {  # Deployment destination mapping
            "production": "main",  # Where code meets consequence
            "staging": "staging",  # The final dress rehearsal
            "development": "develop",  # Where mistakes are still cheap
        },
    },
    "circleci": {
        "version": 2.1,  # API version, not opinion version
        "jobs": {},  # Populated at configuration time
    },
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 📐 PROJECT TEMPLATES & STRUCTURAL ARCHETYPES                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Blueprint topology for rapid project instantiation.
Architectural patterns crystallized into directory structures—why sketch a cabin when you can deploy a citadel? 🏗️
"""


class TemplateStructure(TypedDict):
    """Structure definition for project templates."""

    structure: List[str]  # Directory and file paths
    description: str  # Template purpose and characteristics
    entry_point: Optional[str]  # Primary execution path


# Core architectural patterns—digital foundations for specific domains
PROJECT_TEMPLATES: Dict[str, TemplateStructure] = {
    # ┌────────────────────────────────────────────────────────┐
    # │ Python Library — Reusable Code Artifacts               │
    # └────────────────────────────────────────────────────────┘
    "python_lib": {
        "description": "Modular Python package with testing harness and documentation scaffold.",
        "entry_point": "{project_name}/core.py",
        "structure": [
            "{project_name}/",  # Core package namespace
            "{project_name}/__init__.py",  # Package identity and exports
            "{project_name}/core.py",  # Functional heart of the system
            "tests/",  # Assumption validation chamber
            "tests/__init__.py",  # Test discovery marker
            "tests/test_core.py",  # Core functionality verification
            "docs/",  # Knowledge crystallization zone
            "docs/conf.py",  # Documentation configuration
            "docs/index.rst",  # Entry point for human comprehension
            "README.md",  # First impressions matter
            "CHANGELOG.md",  # Historical record of evolution
            "pyproject.toml",  # Modern build system definition
            "setup.py",  # Legacy installation compatibility
            "LICENSE",  # Legal boundary definition
            ".gitignore",  # Version control exclusion rules
        ],
    },
    # ┌────────────────────────────────────────────────────────┐
    # │ FastAPI Service — High-Performance API Framework       │
    # └────────────────────────────────────────────────────────┘
    "fastapi_service": {
        "description": "Type-driven API service with domain separation and containerization.",
        "entry_point": "{project_name}/main.py",
        "structure": [
            "{project_name}/",  # Service root namespace
            "{project_name}/__init__.py",  # Package identification
            "{project_name}/main.py",  # Application entrypoint
            "{project_name}/api/",  # Interface definition layer
            "{project_name}/api/__init__.py",  # API package marker
            "{project_name}/api/endpoints/",  # Route implementations
            "{project_name}/core/",  # Domain-agnostic utilities
            "{project_name}/models/",  # Data persistence schemas
            "{project_name}/schemas/",  # API contract definitions
            "{project_name}/services/",  # Business logic encapsulation
            "tests/",  # Validation and verification
            "docker/",  # Containerization artifacts
            "docs/",  # API documentation
            ".env.example",  # Configuration template
        ],
    },
    # ┌────────────────────────────────────────────────────────┐
    # │ React Application — Component-Based UI System          │
    # └────────────────────────────────────────────────────────┘
    "react_app": {
        "description": "Modern React application with strict TypeScript and domain separation.",
        "entry_point": "src/index.tsx",
        "structure": [
            "src/",  # Source code enclosure
            "src/components/",  # Reusable UI primitives
            "src/pages/",  # Route-specific compositions
            "src/hooks/",  # Behavior extraction layer
            "src/services/",  # External system interfaces
            "src/styles/",  # Visual language definitions
            "src/assets/",  # Static resource collection
            "public/",  # Unprocessed web assets
            "package.json",  # Dependency declaration
            "tsconfig.json",  # TypeScript compiler rules
            ".eslintrc.js",  # Code quality enforcement
            "jest.config.js",  # Test runner configuration
        ],
    },
    # ┌────────────────────────────────────────────────────────┐
    # │ Node Service — JavaScript Backend Application          │
    # └────────────────────────────────────────────────────────┘
    "node_service": {
        "description": "Express-based Node.js service with MVC architecture.",
        "entry_point": "src/index.js",
        "structure": [
            "src/",  # Source domain boundary
            "src/index.js",  # Application bootstrap
            "src/routes/",  # Request routing definitions
            "src/controllers/",  # Request handler implementations
            "src/models/",  # Data access and validation
            "src/middleware/",  # Request processing pipeline
            "src/utils/",  # Cross-cutting utilities
            "tests/",  # Automated verification
            "package.json",  # Project manifest
            "README.md",  # Documentation front matter
            ".env.example",  # Environment configuration template
            ".gitignore",  # Repository exclusion patterns
        ],
    },
}


# Extension logic for templating engines
def get_template_structure(template_name: str) -> List[str]:
    """
    Retrieve the file structure for a specified template.

    Args:
        template_name: Identifier for the requested template

    Returns:
        Ordered list of files and directories for the template

    Raises:
        KeyError: When requested template doesn't exist in the library

    Examples:
        >>> structure = get_template_structure("python_lib")
        >>> for path in structure:
        ...     print(f"Creating: {path}")
    """
    if template_name not in PROJECT_TEMPLATES:
        available = ", ".join(PROJECT_TEMPLATES.keys())
        raise KeyError(
            f"Template '{template_name}' not found. Available templates: {available}"
        )

    return PROJECT_TEMPLATES[template_name]["structure"]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 📚 DOCUMENTATION STANDARDS & KNOWLEDGE CRYSTALLIZATION                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Precise specification of knowledge transfer mechanisms across mediums.
Good code shows what happens; good docs reveal why anyone should care. 📖
"""


# ┌────────────────────────────────────────────────────────┐
# │ Type Definitions — Contracts for Documentation Config  │
# └────────────────────────────────────────────────────────┘
class SphinxConfig(TypedDict):
    """Sphinx documentation engine configuration."""

    theme: str  # Visual presentation framework
    extensions: List[str]  # Functional augmentation modules


class DocstringConfig(TypedDict):
    """Function-level documentation parameters."""

    style: Literal["google", "numpy", "sphinx"]  # Structural dialect
    required_sections: List[str]  # Mandatory knowledge components


class ChangelogConfig(TypedDict):
    """Historical record categorization."""

    categories: List[str]  # Evolution taxonomy


class ReadmeConfig(TypedDict):
    """Entry point documentation structure."""

    sections: List[str]  # Progressive disclosure sequence


class APIReferenceConfig(TypedDict):
    """Machine-consumable interface documentation."""

    auto_generate: bool  # Automation preference
    include_private: bool  # Exposure boundary


class DocumentationStandardsConfig(TypedDict):
    """Complete documentation topology."""

    sphinx: SphinxConfig
    docstrings: DocstringConfig
    changelog: ChangelogConfig
    readme_sections: List[str]
    api_reference: APIReferenceConfig


# Digital knowledge architecture—precise specifications for thought transmission
DOCUMENTATION_STANDARDS: DocumentationStandardsConfig = {
    "sphinx": {
        "theme": "furo",  # Typography that respects your corneas
        "extensions": [
            "sphinx.ext.autodoc",  # Documentation extraction without surgery
            "sphinx.ext.viewcode",  # Show me the implementation, not just promises
            "sphinx.ext.napoleon",  # Imperial formatting without the conquest
            "sphinx.ext.intersphinx",  # Cross-documentation hyperspace tunnels
            "sphinx.ext.autosummary",  # TL;DR generators for the attention-impaired
            "sphinx_autodoc_typehints",  # Types that explain themselves when asked nicely
            "sphinx_copybutton",  # Liberation from manual transcription torture
            "sphinx_design",  # Because brutalist documentation is a war crime
            "myst_parser",  # Markdown: for when reStructuredText feels sadistic
        ],
    },
    "docstrings": {
        "style": "google",  # Preference, not religion—others are merely suboptimal
        "required_sections": [
            "Args/Parameters",  # What goes in—the function's dietary requirements
            "Returns",  # What comes out—the function's contribution to society
            "Raises",  # When things explode—the function's temperamental side
            "Examples",  # Show, don't tell—worth a thousand type annotations
        ],
    },
    "changelog": {
        "categories": [
            "Added",  # Birth announcements for features
            "Changed",  # Puberty for existing functionality
            "Deprecated",  # Midlife crisis warnings
            "Removed",  # Digital obituaries
            "Fixed",  # Bug rehabilitation program
            "Security",  # Patching the castle walls
        ],
    },
    "readme_sections": [
        "Overview",  # "What is this thing?" for the chronically confused
        "Installation",  # "How do I get it?" for the acquisition-oriented
        "Quick Start",  # "Make it work now!" for the pathologically impatient
        "Features",  # "Why should I care?" for the perpetually skeptical
        "Usage",  # "What else can it do?" for the intellectually curious
        "Contributing",  # "How can I help?" for the suspiciously generous
        "License",  # "Can I use this legally?" for the rightfully paranoid
    ],
    "api_reference": {
        "auto_generate": True,  # Because manual documentation is where accuracy goes to die
        "include_private": False,  # Some implementation details deserve their privacy
    },
}

# Knowledge propagation equivalence table—cross-medium translation guide
DOCUMENTATION_EQUIVALENCE: Dict[str, str] = {
    "Code Comment": "Immediate context",  # The whisper in your ear while coding
    "Docstring": "Function contract",  # The promise your function made in writing
    "README": "Project introduction",  # The handshake before collaboration
    "Changelog": "Historical record",  # The archaeological evidence of progress
    "User Guide": "Practical education",  # The apprenticeship in digital form
    "API Reference": "Technical specification",  # The blueprints for integration architects
}

"""
╭──────────────────────────────────────────────────────╮
│ ⚠️ Documentation Philosophy:                         │
│                                                      │
│ "Documentation isn't decoration—it's the difference  │
│  between a tool and a puzzle."                       │
╰──────────────────────────────────────────────────────╯
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 🏷️ VERSIONING & ARTIFACT PROPAGATION                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Precise versioning topology and distribution channel configuration.
Digital archaeology starts with proper stratigraphy—each version layer tells a story. 🧮
"""


class VersionComponents(TypedDict):
    """Atomic elements of semantic versioning."""

    major: int  # Breaking changes—compatibility funeral directors
    minor: int  # Features—backwards-compatible enhancements
    patch: int  # Fixes—digital wound dressings


class GitConfig(TypedDict):
    """Source control versioning parameters."""

    tag_prefix: str  # Decorative prefix for release identifiers
    commit_convention: Literal["conventional", "angular", "emoji"]  # Message taxonomy


class RegistryConfig(TypedDict):
    """Distribution endpoint specifications."""

    repository: str  # Production artifact destination
    repository_test: str  # Pre-release purgatory


VERSIONING: Dict[str, Any] = {
    "scheme": "semantic",  # Numbers with meaning—not just arbitrary increments
    "current": VersionComponents(
        major=0,  # Zero: the traditional number of stable first releases
        minor=1,  # The loneliest positive integer
        patch=0,  # No bugs fixed because perfection shipped initially
    ),
    "init_version": "0.1.0",  # The genesis state—potential without promises
    "pre_release_prefix": "alpha",  # Warning label for the adventurous
    "build_metadata": "",  # Empty canvas awaiting CI/CD fingerprints
    "git": GitConfig(
        tag_prefix="v",  # The traditional honorific for version numbers
        commit_convention="conventional",  # Structure for machines, meaning for humans
    ),
    "pypi": RegistryConfig(
        repository="https://upload.pypi.org/legacy/",  # Where Python packages go to thrive
        repository_test="https://test.pypi.org/legacy/",  # The rehearsal stage for package debuts
    ),
    "npm": {
        "registry": "https://registry.npmjs.org/",  # JavaScript's package purgatory
    },
    "docker": {
        "repository": "ghcr.io/neuroforge",  # Container cathedral
        "tag": "latest",  # The eternal lie of immutability
    },
}

# Version significance taxonomy—what each increment actually means
VERSION_SEMANTICS: Dict[str, str] = {
    "major": "Breaking changes that demand migration rituals",  # Backward compatibility sacrificed for progress
    "minor": "New features that preserve existing contracts",  # Evolution without disruption
    "patch": "Corrections to unintended behaviors",  # Reality alignment operations
    "pre": "Warning labels for the chronically adventurous",  # Here be dragons, but documented ones
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 🎨 BRANDING & VISUAL IDENTITY                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Visual identity codification system with precise electromagnetic wavelength selections.
The difference between memorable and forgettable is often 3 hexadecimal digits. 🖌️
"""


# Type definitions for branding components
class ColorScheme(TypedDict):
    """Precise electromagnetic wavelength selectors."""

    primary: str  # Brand foundation—the backbone of visual identity
    secondary: str  # Supporting actor that never steals the scene
    accent: str  # Attention magnet for critical interactive elements
    background: str  # Canvas upon which all other elements perform
    text: str  # The voice of your content—should never need to shout
    success: str  # The color of victory and task completion
    warning: str  # "Caution: physics at work" notification
    error: str  # The digital equivalent of sirens
    info: str  # "By the way" in visual form


class Typography(TypedDict):
    """Verbal display configuration parameters."""

    heading: str  # For text that needs to establish hierarchy
    body: str  # For text that actually delivers value
    code: str  # For when monospace is the only space


class AssetLocations(TypedDict):
    """Digital artifact cartography."""

    main: str  # The full regalia
    icon: str  # The minimalist representation
    favicon: str  # The "remember me in your tabs" badge


class SocialPresence(TypedDict):
    """Cross-platform visual representation system."""

    banner: str  # The billboard of your digital estate
    preview: str  # The first impression algorithm food


class BrandingSystem(TypedDict):
    """Complete visual identity specification."""

    colors: ColorScheme
    fonts: Typography
    logos: AssetLocations
    social: SocialPresence
    tagline: str  # The verbal distillation of existence


# The crystallization of visual identity—digital DNA expressed through pixels
BRANDING: BrandingSystem = {
    "colors": {
        "primary": "#2D3E50",  # Deep blue-grey: wisdom without pretension
        "secondary": "#3498DB",  # Clear blue: trust without naivety
        "accent": "#E74C3C",  # Vibrant red: urgency without panic
        "background": "#ECF0F1",  # Light grey: canvas without distraction
        "text": "#2C3E50",  # Dark blue-grey: legibility without strain
        "success": "#2ECC71",  # Green: achievement without arrogance
        "warning": "#F39C12",  # Amber: caution without paralysis
        "error": "#E74C3C",  # Red: failure with dignity
        "info": "#3498DB",  # Blue: knowledge without overload
    },
    "fonts": {
        "heading": "Montserrat, sans-serif",  # Geometric clarity with human touch
        "body": "Open Sans, sans-serif",  # Legible at all sizes, like truth
        "code": "Fira Code, monospace",  # Where ligatures meet logic
    },
    "logos": {
        "main": "assets/logos/eidosian_logo.png",  # Full narrative in pixels
        "icon": "assets/logos/eidosian_icon.svg",  # Vectorized simplicity—infinitely scalable
        "favicon": "assets/logos/favicon.ico",  # 16x16px of instant recognition
    },
    "social": {
        "banner": "assets/social/banner.png",  # Rectangle of first impressions
        "preview": "assets/social/preview.png",  # Link sharing thumbnail—judge-me-by size
    },
    "tagline": "Brutal Clarity, Unmistakably Eidosian.",  # Our promise, distilled
}

# The color wheel isn't a suggestion—it's a precision instrument (ʘ‿ʘ)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 🔒 SECURITY & CRYPTOGRAPHIC ARCHITECTURE                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Cryptographic fortification topology for digital asset protection.
Security isn't a feature—it's the foundation beneath all other foundations. 🛡️
"""


# ┌────────────────────────────────────────────────────────┐
# │ Type Definitions — Contracts for Security Parameters   │
# └────────────────────────────────────────────────────────┘
class EncryptionConfig(TypedDict):
    """Symmetric and asymmetric encryption parameters."""

    algorithm: str  # Cipher selection for data transformation
    key_derivation: str  # Function to transmute passwords into keys
    iterations: int  # Computational work factor against brute force


class HashingConfig(TypedDict):
    """One-way transformation configuration."""

    algorithm: str  # Primary digest generation algorithm
    rounds: int  # Computational expense coefficient
    fallback_algorithms: List[str]  # Backup strategies for legacy verification


class JWTConfig(TypedDict):
    """JSON Web Token authentication parameters."""

    algorithm: str  # Signature algorithm selection
    expiration_minutes: int  # Token lifetime before digital death
    refresh_expiration_days: int  # Regeneration permission duration


class SecretsConfig(TypedDict):
    """Sensitive value storage strategy."""

    provider: str  # Vault technology selection
    mount_point: str  # Logical access path in secret topology


class MFAConfig(TypedDict):
    """Multi-factor authentication parameters."""

    enabled: bool  # Gate control for additional verification
    methods: List[str]  # Authentication channel diversity


class SecurityConfig(TypedDict):
    """Complete security topology."""

    encryption: EncryptionConfig
    hashing: HashingConfig
    jwt: JWTConfig
    secrets_management: SecretsConfig
    two_factor_auth: MFAConfig


# ┌────────────────────────────────────────────────────────┐
# │ Security Configuration — Digital Fort Knox             │
# └────────────────────────────────────────────────────────┘
SECURITY: SecurityConfig = {
    "encryption": {
        "algorithm": "AES-256-GCM",  # The symmetric algorithm that sleeps well at night
        "key_derivation": "PBKDF2",  # Password stretching that would make taffy jealous
        "iterations": 100000,  # Brute force deterrent—like speed bumps for attackers
    },
    "hashing": {
        "algorithm": "bcrypt",  # Password protection with a superiority complex
        "rounds": 12,  # Computational cost—expensive by design
        "fallback_algorithms": [
            "argon2id",  # For when memory-hardness matters more than your deadline
            "pbkdf2_sha256",  # The digital equivalent of a participation trophy
        ],
    },
    "jwt": {
        "algorithm": "HS256",  # HMAC with SHA-256—the Toyota Corolla of signatures
        "expiration_minutes": 60,  # Digital paperwork that self-destructs
        "refresh_expiration_days": 7,  # The statute of limitations for trust
    },
    "secrets_management": {
        "provider": "vault",  # Because environment variables are just .env files in fancy clothes
        "mount_point": "eidosian",  # Our digital safe's combination dial
    },
    "two_factor_auth": {
        "enabled": True,  # Because passwords alone are like houses with front doors only
        "methods": [
            "sms",  # Security via telecommunications oligopolies
            "email",  # The "I have access to another password" verification
            "auth_app",  # Time-based tokens—where math meets security
        ],
    },
}

"""
╭──────────────────────────────────────────────────────╮
│ ⚠️ Security Philosophy:                              │
│                                                      │
│ "The perfect security system is perfectly useless    │
│  if humans can't use it correctly."                  │
╰──────────────────────────────────────────────────────╯
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 📊 UTILITIES                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
Digital surgery tools—precision instruments for information manipulation.
Every data structure eventually requires extraction, transformation, or navigation. 🧰
"""

T = TypeVar("T")  # Type parameter—polymorphism's silent contract

# ┌────────────────────────────────────────────────────────┐
# │ Information Extraction & Presentation                  │
# └────────────────────────────────────────────────────────┘


def print_info(section: Optional[str] = None, format_json: bool = True) -> None:
    """
    Extract and display configuration sections with surgical precision.

    Performs targeted information retrieval from the global configuration
    topology, rendering structured data with optimal cognitive alignment.

    Args:
        section: The section identifier to display (case-insensitive)
        format_json: Whether to apply pretty-printing to structured data

    Returns:
        None - Output materialized to stdout

    Raises:
        KeyboardInterrupt: When users reject their digital destiny

    Examples:
        >>> print_info("AUTHOR_INFO")  # Targeted information extraction
        >>> print_info()  # Full digital terrain survey
    """
    # Type-specific section mapping with precise structural boundaries
    sections: Dict[str, Any] = {
        "SYSTEM_INFO": SYSTEM_INFO,  # Computational context coordinates
        "AUTHOR_INFO": AUTHOR_INFO,  # Attribution topology markers
        "ENVIRONMENT": {  # Execution domain parameters
            "ROOT_DIR": str(ROOT_DIR),  # Truth anchor coordinates
            "PYTHON_VERSION": PYTHON_VERSION,  # Syntax compatibility bounds
            "ENV_CONFIGS": ENV_CONFIGS,  # Reality-state dependent flags
        },
        "BRANDING": BRANDING,  # Visual identity matrix
        "VERSIONING": VERSIONING,  # Temporal position vectors
    }

    if section:
        # Targeted information extraction—surgical precision over carpet bombing
        section_key: str = section.upper()
        if section_key not in sections:
            available: str = ", ".join(sorted(sections.keys()))
            print(f"❌ Section '{section}' not found. Available sections: {available}")
            return

        data: (
            Dict[str, str]
            | Dict[str, str | bool]
            | Dict[str, str | Dict[str, str | bool]]
            | BrandingSystem
            | Dict[
                str,
                str | VersionComponents | GitConfig | RegistryConfig | Dict[str, str],
            ]
        ) = sections[section_key]
        print(f"📋 {section_key}:")
        if format_json:
            # Type-aware rendering based on structural characteristics
            pprint.pprint(
                data,
                indent=2,
                width=100,
                sort_dicts=True,  # Deterministic rendering for reproducible cognition
                compact=False,  # Information density calibrated for human parsing
            )
        else:
            # Raw output for maximum fidelity
            print(data)
    else:
        # Complete cartographic analysis—digital terrain reconnaissance
        horizontal_rule: str = "═" * 80
        for section_name, data in sorted(sections.items()):
            # Visual demarcation enhances cognitive segmentation
            print(f"\n{horizontal_rule}\n📋 {section_name}:\n{horizontal_rule}")
            if format_json:
                # Type-aware rendering based on structural characteristics
                pprint.pprint(
                    data,
                    indent=2,
                    width=100,
                    sort_dicts=True,  # Alphabetical ordering balances chaos
                    compact=False,  # Prefer clarity over compression
                )
            else:
                # Raw output when formatting is disabled
                print(data)


# ┌────────────────────────────────────────────────────────┐
# │ Version Management & Dependency Coordination           │
# └────────────────────────────────────────────────────────┘


def get_version() -> str:
    """
    Retrieve the current semantic version string.

    Returns:
        Formatted version string following semantic versioning principles

    Examples:
        >>> get_version()
        '0.1.0'
        >>> # With pre-release and build metadata:
        >>> # '0.1.0-alpha+20240521'
    """
    v: VersionComponents = VERSIONING["current"]
    version: str = f"{v['major']}.{v['minor']}.{v['patch']}"  # Core version trinity

    # Optional qualifiers—digital identity refinement
    prefix: str = VERSIONING["pre_release_prefix"]
    if prefix:
        version += f"-{prefix}"  # Stability disclaimer

    metadata: str = VERSIONING["build_metadata"]
    if metadata:
        version += f"+{metadata}"  # Provenance marker

    return version


def get_dependency_group(group_name: str) -> List[str]:
    """
    Extract a specific dependency group by name.

    Args:
        group_name: Identifier for the dependency collection

    Returns:
        List of package specifications with version constraints

    Raises:
        KeyError: When requested group doesn't exist in the dependency topology

    Examples:
        >>> get_dependency_group("core")
        ['requests>=2.31.0', 'aiohttp>=3.11.0', ...]
    """
    if group_name not in FULL_DEPENDENCY_MAP:
        available: str = ", ".join(sorted(FULL_DEPENDENCY_MAP.keys()))
        raise KeyError(
            f"Dependency group '{group_name}' not found. "  # Failure should educate
            f"Available groups: {available}"
        )

    return FULL_DEPENDENCY_MAP[group_name]


# ┌────────────────────────────────────────────────────────┐
# │ Environment & Path Resolution                          │
# └────────────────────────────────────────────────────────┘


def get_environment_config(env: str) -> Dict[str, Union[str, bool]]:
    """
    Retrieve configuration for a specific execution environment.

    Args:
        env: Environment identifier (development, testing, etc.)

    Returns:
        Configuration parameters for the specified environment

    Raises:
        KeyError: When requested environment doesn't exist in configuration

    Examples:
        >>> get_environment_config("development")
        {'debug': True, 'log_level': 'DEBUG', 'profile': 'dev'}
    """
    if env not in ENV_CONFIGS:
        available: str = ", ".join(sorted(ENV_CONFIGS.keys()))
        raise KeyError(
            f"Environment '{env}' not defined. "  # The universe you seek doesn't exist
            f"Available environments: {available}"
        )

    return ENV_CONFIGS[env]  # Configuration slice for requested reality


def resolve_paths(*path_parts: str) -> Path:
    """
    Construct an absolute path relative to the project root.

    Args:
        *path_parts: Path components to join with the root directory

    Returns:
        Absolute Path object for the requested location

    Examples:
        >>> resolve_paths("config", "settings.json")
        PosixPath('/home/lloyd/repos/config/settings.json')
    """
    if not path_parts:
        return ROOT_DIR  # Default to universal origin point

    return ROOT_DIR.joinpath(*path_parts)  # Coordinate composition via root anchoring


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 🧬 MODULE IDENTITY & EXPORT TOPOLOGY                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""
Digital fingerprinting for package identification and distribution.
Every byte cluster needs a name, every module deserves proper attribution. 📦
"""

# Core identity markers—immutable truths about this digital artifact
__title__: str = "eidos global info"
__description__: str = "Centralized configuration for all Eidosian repos and projects."
__version__: str = get_version()  # Single source of truth
__author__: str = AUTHOR_INFO["name"]
__email__: str = AUTHOR_INFO["email"]
__license__: str = "MIT"
__copyright__: str = f"Copyright (c) {date.today().year} {AUTHOR_INFO['org']}"

# Export manifest—the complete taxonomy of importable artifacts
# This digital contract specifies exactly what others may access
__all__: List[str] = [
    # Identity & Legal Framework
    "AUTHOR_INFO",  # Human attribution coordinates
    "SYSTEM_INFO",  # Machine context parameters
    "MIT_LICENSE",  # Freedom with attribution
    "APACHE_LICENSE",  # Corporate-friendly permissions
    "GPL_LICENSE",  # Viral freedom propagation
    "BSD_LICENSE",  # Simplified legal boundaries
    "CC_LICENSE",  # Content distribution terms
    # Filesystem Cartography
    "ROOT_DIR",  # The origin point of truth
    "VENV_DIR",  # Dependency isolation chamber
    "DATA_DIR",  # Information transformation zone
    "CACHE_DIR",  # Ephemeral access optimization
    "LOGS_DIR",  # Execution history archives
    "CONFIG_DIR",  # Behavioral parameter storage
    "TEMP_DIR",  # Transient workspace coordinates
    "ASSETS_DIR",  # Static resource repository
    "MODELS_DIR",  # Crystallized intelligence store
    "DOCS_BUILD_DIR",  # Knowledge rendering output
    # Version Parameters
    "PYTHON_VERSION",  # Interpreter compatibility bounds
    "NODE_VERSION",  # JavaScript runtime requirements
    "GO_VERSION",  # Golang compiler constraints
    "RUST_VERSION",  # Safety-first language markers
    # Configuration Domains
    "ENV_CONFIGS",  # Context-aware execution parameters
    # Package Ecosystems
    "DEV_DEPENDENCIES",  # Creation toolchain components
    "DOCS_DEPENDENCIES",  # Documentation substrate packages
    "CORE_DEPENDENCIES",  # Foundational runtime libraries
    "DATA_DEPENDENCIES",  # Information processing arsenal
    "ML_DEPENDENCIES",  # Intelligence augmentation kit
    "WEB_DEPENDENCIES",  # Network presence framework
    "CLOUD_DEPENDENCIES",  # Distributed infrastructure connectors
    # Architectural Components
    "CONTAINERIZATION",  # Deployment isolation strategy
    "CODE_STYLE",  # Aesthetic consistency rules
    "TESTING_CONFIGS",  # Validation methodology specs
    "CICD_CONFIGS",  # Automation pipeline settings
    "PROJECT_TEMPLATES",  # Architectural blueprints
    "DOCUMENTATION_STANDARDS",  # Knowledge transfer protocols
    "VERSIONING",  # Change tracking specifications
    "BRANDING",  # Visual identity parameters
    "SECURITY",  # Protection mechanism settings
]

"""
Global information and initialization for Figlet Forge.

This module contains global constants and initialization functions
for the Figlet Forge system, ensuring proper configuration and setup.
"""

# Ensure src directory is in Python path
SRC_DIR = Path(__file__).parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Package information
PACKAGE_NAME = "figlet_forge"
PACKAGE_VERSION = "1.0.2"
PACKAGE_AUTHOR = "Lloyd Handyside"
PACKAGE_EMAIL = "ace1928@gmail.com"
PACKAGE_DESCRIPTION = (
    "Enhanced Figlet System with colorized ANSI support and Unicode rendering"
)

# Font directories to search
FONT_DIRS: List[Path] = [
    Path.home() / ".figlet_forge" / "fonts",
    Path("/usr/share/figlet"),
    Path("/usr/local/share/figlet"),
]

# Add package fonts directory
try:
    import figlet_forge

    PACKAGE_DIR = Path(figlet_forge.__file__).parent
    FONT_DIRS.append(PACKAGE_DIR / "fonts")
except ImportError:
    pass

# Environment variable for font path
if "FIGLET_FORGE_FONT_DIR" in os.environ:
    font_dir = Path(os.environ["FIGLET_FORGE_FONT_DIR"])
    if font_dir.exists():
        FONT_DIRS.insert(0, font_dir)


# Function to find available fonts
def get_available_fonts() -> Set[str]:
    """
    Find all available figlet fonts in the system.

    Returns:
        Set of available font names
    """
    fonts: Set[str] = set()

    for font_dir in FONT_DIRS:
        if font_dir.exists():
            for font_file in font_dir.glob("*.flf"):
                fonts.add(font_file.stem)

    return fonts


# Initialize package imports if needed
def ensure_package_imports() -> None:
    """
    Ensure figlet_forge package can be imported.

    This function is used for development environments to make sure
    the package is importable without installation.
    """
    if "figlet_forge" not in sys.modules:
        try:
            import figlet_forge  # noqa
        except ImportError:
            if SRC_DIR.exists():
                sys.path.insert(0, str(SRC_DIR))
