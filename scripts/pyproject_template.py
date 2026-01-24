from eidosian_core import eidosian

@eidosian()
def get_pyproject_content(name):
    return f"""[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
description = "Eidosian Forge Component: {name}"
readme = "README.md"
authors = [
  {{ name = "Eidosian Nexus", email = "eidos@neuroforge.io" }},
]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[tool.setuptools.packages.find]
where = ["src"]
"""
