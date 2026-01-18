import sys
import types
from pathlib import Path
from typing import Any, Literal

# --- Path Configuration and Module Resolution -----------------------------------
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Ensure type_forge and BaseValidator are properly available
type_forge_path = project_root / "src" / "type_forge"
if type_forge_path.exists():
    sys.path.insert(0, str(type_forge_path))

    # Pre-emptive module creation to ensure BaseValidator availability
    if "type_forge.core.base" not in sys.modules:
        # Create BaseValidator placeholder to prevent KeyError
        base_module = types.ModuleType("type_forge.core.base")
        sys.modules["type_forge.core.base"] = base_module

        # Define BaseValidator placeholder with essential attributes
        class BaseValidatorPlaceholder:
            """Base class for all validators in the Type Forge system."""

            def __init__(self):
                """Initialize the BaseValidator."""
                pass

            def validate(self, value: Any) -> Any:
                """Validate method placeholder."""
                return value

        # Register the BaseValidator placeholder
        base_module.BaseValidator = BaseValidatorPlaceholder

    # Try to import actual implementation if available
    try:
        import type_forge.core.base
    except ImportError:
        pass  # Will use the placeholder created above

# --- Project Information -------------------------------------------------------
project = "Type Forge"
copyright = "2025, Lloyd Handyside"
author = "Lloyd Handyside"
release = "0.1.0"

# --- MyST Compatibility Settings -----------------------------------------------
suppress_warnings = ["myst.footnote", "ref.python", "toc.not_included", "autoapi.*"]
myst_footnote_transition = True
myst_dmath_double_inline = True

# --- Extensions Configuration --------------------------------------------------
extensions = [
    # Core documentation
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # Markdown support
    "myst_parser",
    "myst_nb",
    # Auto API documentation (explicitly first)
    "autoapi.extension",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autodoc_pydantic",
    # Enhanced features
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinxext.opengraph",
    "sphinxcontrib.mermaid",
    # Cross-referencing
    "sphinx.ext.autosectionlabel",
]

# --- AutoAPI Configuration -----------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../src"]
autoapi_template_dir = "_templates/autoapi"
autoapi_add_toctree_entry = True

# Enhanced import and members handling
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
    "special-members",
]

# Advanced namespace handling
autoapi_python_use_implicit_namespaces = True
autoapi_keep_files = True
autoapi_ignore_local_modules = False
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_file_patterns = ["*.py", "*.pyi"]
autoapi_generate_api_docs = True

# Special handling for BaseValidator
autoapi_modules = {
    "type_forge.core.base": {
        "override": True,
        "prune": False,
        "private": False,
    },
}

# --- Documentation Format Settings ---------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",
}

# MyST markdown extensions
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 4  # Generate anchors for h1-h4

# --- Napoleon Documentation Style Settings -------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = ["Attributes", "Methods", "Examples"]

# --- HTML Output Configuration -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_favicon = "_static/favicon.ico"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# RTD theme options
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
    "style_external_links": True,
    "prev_next_buttons_location": "bottom",
}

# --- Additional Sphinx Configuration -------------------------------------------
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3

# Typehint rendering
autodoc_typehints: Literal["description"] = "description"
autodoc_typehints_format: Literal["short"] = "short"
typehints_fully_qualified = False
typehints_document_rtype = True

# --- Type Variables and Reference Handling -------------------------------------
# Base type variables
_base_type_vars = [
    ("py:class", "T"),
    ("py:class", "S"),
    ("py:class", "R"),
    ("py:class", "K"),
    ("py:class", "V"),
    ("py:class", "U"),
    ("py:class", "HashableT"),
    ("py:class", "SchemaTypeT"),
    ("py:class", "T_contra"),
    ("py:class", "U_co"),
    ("py:class", "T_co"),
    ("py:class", "S_contra"),
]

# Domain-specific types
_domain_types = [
    ("py:class", "TError"),
    ("py:class", "TInstance"),
    ("py:class", "TypeName"),
    ("py:class", "ErrorMessage"),
    ("py:class", "ValidationPath"),
    ("py:class", "TypeDistance"),
    ("py:class", "ValidationOptions"),
    ("py:class", "TypeViolation"),
    ("py:class", "ValidationResult"),
    ("py:class", "ConversionResult"),
    ("py:class", "FieldDefinitions"),
    ("py:class", "SchemaValueT"),
    ("py:class", "TypeRegistry"),
    ("py:class", "ParentSpecType"),
    ("py:class", "Ellipsis"),
    ("py:class", "TypeCategoryLiteral"),
    ("py:class", "TypeObject"),
    ("py:class", "optional"),
]

# Exception handling
_exception_types = [
    ("py:exc", "No direct exceptions"),
    ("py:exc", "but captures exceptions from validate"),
    ("py:exc", "and"),
    ("py:exc", "converts them to ValidationResult with appropriate violations"),
]

# Module-specific type variables
_type_forge_types = [
    # Core type variables
    ("py:class", "type_forge.typing.definitions.T"),
    ("py:class", "type_forge.typing.definitions.S"),
    ("py:class", "type_forge.typing.definitions.R"),
    ("py:class", "type_forge.typing.definitions.U"),
    ("py:class", "type_forge.typing.definitions.K"),
    ("py:class", "type_forge.typing.definitions.V"),
    ("py:class", "type_forge.typing.definitions.HashableT"),
    # Typing variables
    ("py:class", "type_forge.typing.variables.T"),
    ("py:class", "type_forge.typing.variables.S"),
    ("py:class", "type_forge.typing.variables.R"),
    ("py:class", "type_forge.typing.variables.U"),
    ("py:class", "type_forge.typing.variables.T_co"),
    ("py:class", "type_forge.typing.variables.S_contra"),
    ("py:class", "type_forge.typing.variables.T_contra"),
    ("py:class", "type_forge.typing.variables.U_co"),
]

# BaseValidator references
_base_validator_refs = [
    # All possible reference types for BaseValidator
    ("py:class", "type_forge.core.base.BaseValidator"),
    ("py:obj", "type_forge.core.base.BaseValidator"),
    ("py:mod", "type_forge.core.base.BaseValidator"),
    ("py:exc", "type_forge.core.base.BaseValidator"),
    ("py:attr", "type_forge.core.base.BaseValidator"),
    ("py:func", "type_forge.core.base.BaseValidator"),
    ("py:meth", "type_forge.core.base.BaseValidator"),
    # Src-prefixed paths
    ("py:class", "src.type_forge.core.base.BaseValidator"),
    ("py:obj", "src.type_forge.core.base.BaseValidator"),
    # Alternative formats
    ("py:class", "BaseValidator"),
    ("py:obj", "BaseValidator"),
]

# Combine all nitpick ignore entries
nitpick_ignore = (
    _base_type_vars
    + _domain_types
    + _exception_types
    + _type_forge_types
    + _base_validator_refs
)

# Deduplicate entries
nitpick_ignore = list(set(nitpick_ignore))

# --- Cross References and External Links ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}

# Link checking configuration
linkcheck_ignore = ["CONTRIBUTING.md"]

# --- Final Sphinx Configuration ------------------------------------------------
primary_domain = "py"
default_role = "py:obj"
master_doc = "index"
myst_update_mathjax = False

# Default autodoc options
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "imported-members": True,
}


# --- Custom Setup for Sphinx Application ---------------------------------------
def setup(app):
    """Enhanced setup to ensure BaseValidator is properly registered.

    This function creates a robust registration for BaseValidator to prevent
    KeyError issues during documentation building.

    Args:
        app: The Sphinx application instance
    """
    # Create type_forge core modules in sys.modules if needed
    module_paths = [
        "type_forge",
        "type_forge.core",
        "type_forge.core.base",
    ]

    # Ensure all module paths exist
    for path in module_paths:
        if path not in sys.modules:
            sys.modules[path] = types.ModuleType(path)

    # Define BaseValidator if needed
    if not hasattr(sys.modules["type_forge.core.base"], "BaseValidator"):
        # Create a placeholder BaseValidator class with basic methods
        class BaseValidatorPlaceholder:
            """Base class for all type validators."""

            def validate(self, value: Any) -> Any:
                """Placeholder validation method."""
                return value

        # Register the placeholder
        sys.modules["type_forge.core.base"].BaseValidator = BaseValidatorPlaceholder

    # Register BaseValidator with autodoc
    app.connect(
        "autodoc-skip-member",
        lambda app, what, name, obj, skip, options: (
            False if name == "BaseValidator" else skip
        ),
    )

    # Add object type
    app.add_object_type(
        "validator",
        "validator",
        objname="validator",
        indextemplate="pair: %s; validator",
    )

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
