#!/usr/bin/env python3
"""Generate setup.py from pyproject.toml."""
from __future__ import annotations

from pathlib import Path
import argparse
import textwrap

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
SETUP_PY = ROOT / "setup.py"


def load_pyproject() -> dict:
    if not PYPROJECT.exists():
        raise FileNotFoundError(f"Missing {PYPROJECT}")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def render_setup(data: dict) -> str:
    project = data.get("project", {})
    build = data.get("build-system", {})
    name = project.get("name", "glyph_forge")
    version = project.get("version", "0.0.0")
    description = project.get("description", "")
    readme = project.get("readme", "README.md")
    requires_python = project.get("requires-python", "")
    license_str = project.get("license", "")
    authors = project.get("authors", [])
    keywords = project.get("keywords", [])
    classifiers = project.get("classifiers", [])
    dependencies = project.get("dependencies", [])
    optional = project.get("optional-dependencies", {})
    scripts = project.get("scripts", {})

    author = authors[0].get("name") if authors else ""
    author_email = authors[0].get("email") if authors else ""

    console_scripts = [f"{k}={v}" for k, v in scripts.items()]

    return textwrap.dedent(
        f'''\
        #!/usr/bin/env python3
        """Auto-generated setup.py from pyproject.toml.

        Do not edit manually. Run: python scripts/sync_setup.py --write
        """
        from pathlib import Path
        from setuptools import find_packages, setup

        ROOT = Path(__file__).resolve().parent
        README = ROOT / "{readme}"

        def read_long_description() -> str:
            if README.exists():
                return README.read_text(encoding="utf-8")
            return "{description}"

        setup(
            name="{name}",
            version="{version}",
            description="{description}",
            long_description=read_long_description(),
            long_description_content_type="text/markdown",
            author="{author}",
            author_email="{author_email}",
            license="{license_str}",
            python_requires="{requires_python}",
            package_dir={{"": "src"}},
            packages=find_packages(where="src"),
            include_package_data=True,
            install_requires={dependencies!r},
            extras_require={optional!r},
            entry_points={{"console_scripts": {console_scripts!r}}},
            keywords={keywords!r},
            classifiers={classifiers!r},
        )
        '''
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write setup.py")
    args = parser.parse_args()

    data = load_pyproject()
    rendered = render_setup(data)

    if args.write:
        SETUP_PY.write_text(rendered, encoding="utf-8")
        print(f"Wrote {SETUP_PY}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
