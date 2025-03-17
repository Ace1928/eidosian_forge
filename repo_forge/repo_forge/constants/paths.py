"""
Path constants for Eidosian repository structure.

This module defines the standard directory structures and paths used
for creating consistent Eidosian monorepo architectures.
"""
from typing import Dict, Any, List

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Core Directory Structure                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Top-level directories that form the foundation of the monorepo
CORE_DIRECTORIES = [
    "projects",      # Language-specific projects
    "libs",          # Shared libraries
    "tools",         # Development and build tools
    "scripts",       # Automation scripts
    "docs",          # Documentation
    "tests",         # Testing framework and shared tests
    "benchmarks",    # Performance benchmarks
    "examples",      # Example code and usage examples
    "ci",            # CI/CD configuration
    ".github",       # GitHub-specific configuration
    "config",        # Configuration files
    "shared",        # Shared cross-language artifacts
    "dist",          # Distribution artifacts
    "build",         # Build artifacts
    "wheelhouse",    # Python wheel packages
]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Language-Specific Directory Structures                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

LANGUAGE_DIRECTORIES = {
    "python": {
        "structure": [
            "core",
            "utils",
            "models",
            "api",
            "services",
            "config",
        ],
        "files": [
            "__init__.py",
            "core/__init__.py",
            "utils/__init__.py",
            "models/__init__.py",
            "api/__init__.py",
            "services/__init__.py",
            "config/__init__.py",
        ],
    },
    "nodejs": {
        "structure": [
            "src",
            "config",
            "api",
            "services",
            "models",
            "utils",
        ],
        "files": [
            "src/index.js",
            "config/index.js",
            "api/index.js",
            "services/index.js",
            "models/index.js",
            "utils/index.js",
        ],
    },
    "go": {
        "structure": [
            "pkg",
            "cmd",
            "internal",
            "api",
            "config",
            "models",
        ],
        "files": [
            "pkg/pkg.go",
            "cmd/main.go",
            "internal/internal.go",
            "api/api.go",
            "config/config.go",
            "models/models.go",
        ],
    },
    "rust": {
        "structure": [
            "src/bin",
            "src/lib",
            "src/api",
            "src/models",
            "src/utils",
            "src/config",
        ],
        "files": [
            "src/lib.rs",
            "src/bin/main.rs",
            "src/api/mod.rs",
            "src/models/mod.rs",
            "src/utils/mod.rs",
            "src/config/mod.rs",
        ],
    },
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Script Directories Structure                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

SCRIPT_DIRECTORIES = [
    "build",         # Build automation scripts
    "deploy",        # Deployment scripts
    "setup",         # Environment setup scripts
    "maintenance",   # System maintenance scripts
    "database",      # Database management scripts
    "utils",         # Utility scripts
    "ci",            # CI/CD helper scripts
    "dev",           # Development utility scripts
]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Test Directories Structure                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

TEST_DIRECTORIES = [
    "unit",          # Unit tests
    "integration",   # Integration tests
    "e2e",           # End-to-end tests
    "performance",   # Performance tests
    "security",      # Security tests
    "fixtures",      # Test fixtures
    "mocks",         # Mock objects
    "utils",         # Test utilities
]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Benchmark Directories Structure                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

BENCHMARK_DIRECTORIES = [
    "performance",   # Performance benchmarks
    "memory",        # Memory usage benchmarks
    "concurrency",   # Concurrency benchmarks
    "io",            # I/O benchmarks
    "network",       # Network benchmarks
    "reports",       # Benchmark reports
    "tools",         # Benchmark tools
]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Example Directories Structure                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

EXAMPLE_DIRECTORIES = [
    "tutorials",     # Step-by-step tutorials
    "quickstart",    # Quick start examples
    "advanced",      # Advanced usage examples
    "integrations",  # Integration examples
    "snippets",      # Code snippets
    "use-cases",     # Real-world use cases
]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ CI Directories Structure                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

CI_DIRECTORIES = [
    "github",        # GitHub Actions configurations
    "gitlab",        # GitLab CI configurations
    "azure",         # Azure DevOps configurations
    "jenkins",       # Jenkins configurations
    "common",        # Shared CI scripts and configurations
]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ GitHub Directories Structure                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

GITHUB_DIRECTORIES = [
    "workflows",     # GitHub Actions workflow files
    "ISSUE_TEMPLATE", # Issue templates
    "PULL_REQUEST_TEMPLATE", # PR templates
    "actions",       # Custom GitHub Actions
]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Documentation Structure                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Align with universal_doc_structure.md
DOCUMENTATION_STRUCTURE = [
    "manual",        # Hand-written documentation
    "auto",          # Auto-generated documentation
    "assets",        # Documentation assets (images, diagrams)
]

# Language-specific manual documentation structure
MANUAL_DOC_STRUCTURE = [
    "guides",           # High-level guides & tutorials
    "api",              # Manually written API documentation
    "design",           # Architectural decisions & overviews
    "examples",         # Example code explanations
    "best_practices",   # Coding conventions & best practices
    "troubleshooting",  # Common issues and resolutions
    "security",         # Security guidelines and considerations
    "changelog",        # Version history & release notes
    "contributing",     # Contribution guidelines
    "faq",              # Frequently Asked Questions
]

# Auto-generated documentation structure
AUTO_DOC_STRUCTURE = [
    "api",              # AutoAPI-generated documentation
    "models",           # Data models documentation
    "functions",        # Function-level documentation
    "error_handling",   # Exception handling documentation
    "benchmarks",       # Performance benchmarks
    "internal",         # Internal API docs
    "schemas",          # Database or data structure schemas
    "configuration",    # Auto-generated config files & references
]

# Assets structure
ASSETS_STRUCTURE = [
    "images",           # Screenshots, diagrams, illustrations
    "diagrams",         # Architecture flowcharts, UML diagrams
    "css",              # Custom stylesheets
    "fonts",            # Custom fonts
]

# Add shared cross-language directories
SHARED_DIRECTORIES = [
    "interfaces",    # Shared interfaces
    "protocols",     # Communication protocols
    "schemas",       # Data schemas
    "types",         # Type definitions
    "protos",        # Protocol buffers
    "tools",         # Shared tools
]

# Configuration directories
CONFIG_DIRECTORIES = [
    "development",   # Development environment configs
    "staging",       # Staging environment configs
    "production",    # Production environment configs
    "testing",       # Testing environment configs
    "local",         # Local environment configs
]

# Tool directories
TOOL_DIRECTORIES = [
    "linters",       # Code linters
    "formatters",    # Code formatters
    "analyzers",     # Code analyzers
    "generators",    # Code generators
    "converters",    # Data converters
]
