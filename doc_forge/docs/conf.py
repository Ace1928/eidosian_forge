#!/usr/bin/env python3
"""
Sphinx Configuration for Documentation
======================================

Welcome to the Eidosian configuration—where structure meets wit and digital brilliance.
This file controls our Sphinx documentation build, engineered to be as robust and universal
as the Eidosian spirit itself. Prepare to witness precision, creativity, and a hint of humour!
"""

import datetime
import importlib  # ensures 'importlib' is defined even if importlib.metadata is missing
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Path Configuration - Crafting our digital landscape
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Determine the absolute path to this documentation directory (our creative studio)
docs_path = Path(__file__).parent.absolute()
# Determine the project root (the base of our magnum opus)
root_path = docs_path.parent

# Ensure source code is discoverable - essential for auto documentation
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))
sys.path.insert(0, str(docs_path))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Project Information - The Heart of Eidosian Brilliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
project = "Doc Forge"
copyright = f"2025-{datetime.datetime.now().year}, MIT License"
author = "Lloyd Handyside"

# Version handling - Because every masterpiece deserves its signature.
try:
    from version_forge.version import (
        get_version_string,  # type: ignore
        get_version_tuple,
    )

    if (root_path / "version.py").exists():
        sys.path.insert(0, str(root_path))
        version = get_version_string()
        release = version
    elif (
        root_path / "src" / project.lower().replace(" ", "_") / "version.py"
    ).exists():
        version = get_version_string()
        release = version
    else:
        try:
            import importlib.metadata

            version = importlib.metadata.version(project.lower().replace(" ", "_"))
            release = version
        except (ImportError, importlib.metadata.PackageNotFoundError):
            version = "0.1.0"
            release = version
except ImportError:
    version = "0.1.0"
    release = version

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# General Configuration - Our Toolkit of Extensions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.todo",
    "sphinx.ext.ifconfig",
]

ext_list = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
    "sphinx_autoapi",
    "sphinx.ext.autosectionlabel",
    "sphinx_sitemap",
    "sphinx_tabs.tabs",
    "sphinx_markdown_tables",
    "notfound.extension",
    "sphinx_inline_tabs",
    "sphinxext.opengraph",
    "sphinx_design",
]

for ext in ext_list:
    try:
        __import__(ext.split(".")[0])
        extensions.append(ext)
    except ImportError:
        pass

autosummary_generate = True
autosummary_imported_members = True

# AutoAPI Configuration - The heart of automated documentation
if "sphinx_autoapi" in extensions:
    autoapi_dirs = [str(root_path)]  # Scan entire repo
    autoapi_type = "python"
    autoapi_outdir = str(docs_path / "auto_docs")
    autoapi_template_dir = (
        str(docs_path / "_templates" / "autoapi")
        if (docs_path / "_templates" / "autoapi").exists()
        else None
    )
    autoapi_options = [
        "members",
        "undoc-members",
        "show-inheritance",
        "show-module-summary",
        "imported-members",
        "special-members",
    ]
    autoapi_python_class_content = "both"
    autoapi_member_order = "groupwise"
    autoapi_file_patterns = ["*.py", "*.pyi"]
    autoapi_generate_api_docs = True
    autoapi_add_toctree_entry = True
    autoapi_keep_files = True

# MyST Parser configuration
if "myst_parser" in extensions:
    myst_enable_extensions = [
        "amsmath",
        "colon_fence",
        "deflist",
        "dollarmath",
        "html_image",
        "linkify",
        "replacements",
        "smartquotes",
        "tasklist",
        "fieldlist",
        "attrs_block",
        "attrs_inline",
    ]
    myst_heading_anchors = 4
    myst_all_links_external = False
    myst_url_schemes = ("http", "https", "mailto", "ftp")

# Napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_attr_annotations = True
napoleon_preprocess_types = True

# Autodoc settings
autodoc_default_options: Dict[str, Union[str, bool]] = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__, __call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": True,
    "private-members": True,
    "ignore-module-all": False,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = True
autodoc_typehints_description_target = "documented"
autodoc_mock_imports = []

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".txt": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    ".git",
    ".gitignore",
    "venv",
    "env",
    ".env",
    ".venv",
    ".github",
    "__pycache__",
    "*.pyc",
]

master_doc = "index"
language = "en"
today_fmt = "%Y-%m-%d"
highlight_language = "python3"
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML Output Configuration - Where Beauty Meets Functionality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
html_theme = "sphinx_rtd_theme" if "sphinx_rtd_theme" in extensions else "alabaster"
html_static_path = ["_static"]
try:
    static_dir = docs_path / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

html_css_files = (
    ["css/custom.css"]
    if (docs_path / "_static" / "css").mkdir(parents=True, exist_ok=True)
    or (docs_path / "_static" / "css").exists()
    else []
)
html_js_files = (
    ["js/custom.js"]
    if (docs_path / "_static" / "js").mkdir(parents=True, exist_ok=True)
    or (docs_path / "_static" / "js").exists()
    else []
)

try:
    css_dir = docs_path / "_static" / "css"
    css_dir.mkdir(parents=True, exist_ok=True)
    custom_css = css_dir / "custom.css"
    if not custom_css.exists():
        with open(custom_css, "w") as f:
            f.write(
                """/* Custom CSS for documentation */
:root {
    --font-size-base: 16px;
    --line-height-base: 1.5;
}

.wy-nav-content {
    max-width: 900px;
}

.highlight {
    border-radius: 4px;
}
"""
            )
except Exception:
    pass

if html_theme == "sphinx_rtd_theme":
    html_theme_options: Dict[str, Union[str, bool, int]] = {
        "prev_next_buttons_location": "bottom",
        "style_external_links": True,
        "style_nav_header_background": "#2980B9",
        "collapse_navigation": False,
        "sticky_navigation": True,
        "navigation_depth": 4,
        "includehidden": True,
        "titles_only": False,
    }

html_title = f"{project} {version} Documentation"
html_short_title = project
html_favicon = (
    "_static/favicon.ico" if (docs_path / "_static" / "favicon.ico").exists() else None
)
html_logo = (
    "_static/logo.png" if (docs_path / "_static" / "logo.png").exists() else None
)

html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_copy_source = True
html_use_index = True
html_split_index = False
html_baseurl = ""

# Sitemap configuration
if "sphinx_sitemap" in extensions:
    sitemap_filename = "sitemap.xml"
    sitemap_url_scheme = "{link}"
    html_baseurl = "https://doc-forge.readthedocs.io/"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LaTeX/PDF Output Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "figure_align": "htbp",
    "preamble": r"""
        \usepackage{charter}
        \usepackage[defaultsans]{lato}
        \usepackage{inconsolata}
    """,
}
latex_documents = [
    (
        master_doc,
        f'{project.lower().replace(" ", "_")}.tex',
        f"{project} Documentation",
        author,
        "manual",
    ),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Additional Output Formats - EPUB, Man, and Texinfo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_show_urls = "footnote"

man_pages = [
    (
        master_doc,
        project.lower().replace(" ", ""),
        f"{project} Documentation",
        [author],
        1,
    )
]

texinfo_documents = [
    (
        master_doc,
        project.lower().replace(" ", "_"),
        f"{project} Documentation",
        author,
        project,
        "Project documentation.",
        "Miscellaneous",
    ),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extension-specific and Miscellaneous Configurations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
todo_include_todos = True
todo_emit_warnings = False
todo_link_only = False

if "sphinx_copybutton" in extensions:
    copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
    copybutton_prompt_is_regexp = True

add_module_names = False
python_use_unqualified_type_names = True

if "sphinxcontrib.mermaid" in extensions:
    mermaid_version = "10.6.1"
    mermaid_init_js = "mermaid.initialize({startOnLoad:true, securityLevel:'loose'});"

if "sphinx.ext.autosectionlabel" in extensions:
    autosectionlabel_prefix_document = True
    autosectionlabel_maxdepth = 3

if "sphinxext.opengraph" in extensions:
    ogp_site_url = "https://doc-forge.readthedocs.io/"
    ogp_image = (
        "_static/logo.png" if (docs_path / "_static" / "logo.png").exists() else None
    )
    ogp_use_first_image = True
    ogp_description_length = 300


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration with doc_forge modules (Eidosian synergy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def integrate_doc_forge(app: Any) -> None:
    """Leverage doc_forge modules for advanced Eidosian doc management."""
    try:
        # Example usage calls (these can be refined as needed)
        from doc_forge.doc_manifest_manager import load_doc_manifest
        from doc_forge.doc_toc_analyzer import analyze_toc
        from doc_forge.doc_validator import validate_docs
        from doc_forge.fix_inline_refs import fix_inline_references
        from doc_forge.update_toctrees import update_toctrees

        # Attempt to update partial docs automatically
        update_toctrees(docs_path)
        fix_inline_references(docs_path)
        analyze_toc(docs_path)
        validate_docs(docs_path)
        load_doc_manifest(root_path / "docs" / "docs_manifest.json")
    except Exception:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# App Setup Hook - Final Flourish and Runtime Customisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup(app: Any) -> Dict[str, Union[str, bool]]:
    """Set up custom configurations for the Sphinx application with Eidosian flair."""
    try:
        app.add_css_file("css/custom.css")

        # Quietly override any conflicting 'tab' directive warnings
        # (to avoid "directive 'tab' is already registered")
        # We keep both tab extensions but skip re-register if needed.
        directive_registry = getattr(app, "registry", None)
        if directive_registry and hasattr(directive_registry, "directives"):
            if "tab" in directive_registry.directives:
                pass  # Already registered, do nothing.

        # Find all Markdown/RST files in the docs directory
        def find_doc_files(docs_dir: Path) -> List[str]:
            all_files = []
            for ext in [".rst", ".md"]:
                for file_path in docs_dir.glob(f"**/*{ext}"):
                    if not any(
                        pattern in str(file_path) for pattern in exclude_patterns
                    ):
                        rel_path = file_path.relative_to(docs_dir)
                        all_files.append(str(rel_path).replace("\\", "/"))
            return all_files

        app.connect("builder-inited", lambda _: find_doc_files(docs_path))

        # Integrate doc_forge modules
        integrate_doc_forge(app)

    except Exception:
        pass

    return {
        "version": version,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
