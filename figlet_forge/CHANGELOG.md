# Changelog

All notable changes to Figlet Forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Unreleased

- Full TypeScript-style type annotations throughout the codebase
- Comprehensive documentation following Eidosian principles
- Enhanced error handling with detailed error messages and recovery suggestions
- HTML and SVG output formats for rendering

### Fixed - Unreleased

- Resolved issues in the color effects module
- Corrected rainbow_colorize function structure
- Fixed documentation formatting in figfont.md

## [0.1.0] - 2023-12-15

### Added - 0.1.0

- Full colorized ANSI code support for vibrant text art
- Unicode character rendering with comprehensive mapping
- Expanded font ecosystem with careful attention to licensing
- Intelligent fallbacks for compatibility with older systems
- Significant performance optimizations without sacrificing quality
- Enhanced maintainability through modern Python practices
- Comprehensive documentation for all use cases
- Full backward compatibility with pyfiglet
- Integrated with the Eidosian Forge ecosystem as a core typography tool
- Modern project structure with pyproject.toml replacing setup.py

#### Technical Improvements - 0.1.0

- Type annotations throughout codebase (mypy, pyright integration)
- Modular architecture for better extensibility
- Unit tests covering all functionality (pytest)
- Code quality enforced via pre-commit hooks
- Documentation built with Sphinx and ReadTheDocs integration

## [0.0.1] - 2023-07-15

### Added - 0.0.1

- Project initialization
- Basic structure and planning
- Core implementation concept
- Compatibility layer design

## Historical pyfiglet Changelog

Below is the changelog from the original pyfiglet project, maintained for historical reference.

### pyfiglet 1.0.2 - 2023-09-13

- Fixed a leaked file descriptor

### pyfiglet 1.0.1 - 2023-09-10

- Added python_requires >= 3.9 to setup.py to prevent old python versions from picking up the newly incompatible version

### pyfiglet 1.0.0 - 2023-09-10 (yanked)

This release was yanked, because it introduced incompatibilities with old
versions of python but did not specify a python_requires line.

- Support for Python 2 was dropped
- Support for compressed fonts
- Support for more fonts
- Various fixes for corner cases of font rendering
- Drop use of pkg_resources in favour of importlib.resources
- Add pyproject.toml
- Many small fixes

### pyfiglet 0.8.0 - 2018-12-06

- #62 Change LICENSE to MIT
- #61 Provide font installation option (-L) and remove
  unlicensed fonts from the distribution

### pyfiglet 0.7.6 - 2018-10-17

- #57 Implement colored print
- #53 Allow fonts to be specified by path

### pyfiglet 0.7.5 - 2016-06-12

- #46 Add 100+ fonts from java.de figlet fonts collection v4.0
- #48 Include python3 in testing

### pyfiglet 0.7.4 - 2015-05-27

- #43 Don't leak file handles

### pyfiglet 0.7.3 - 2015-04-14

- #41 Add newline and text wrapping support

### pyfiglet 0.7.2 - 2014-09-14

- #35 Add this CHANGELOG
- #36 Bug fix for #34 (broken --reverse and --flip)
  (reported "character mapping must return integer, None or unicode")

### pyfiglet 0.7.1 - 2014-07-27

- #29 Fix for UTF8 regression
- #31 Add **main** entry point
- #32 Pep8 the code and minor refactoring
- #33 Trove classifiers update

### pyfiglet 0.7 - 2014-06-02

- #9 Add --list_fonts and --info_font
- #10 Add tools/pyfiglet_example for listing all fonts
- #11 Fix the pyfiglet command (had bad python path)
- #12 Pyflakes fixes
- #13 Configure Travis Continuous Integration
- #17 Documentation usage sample fix
- #19 Enable pyfiglet to use extended ASCII chars
- #20 Add two cyrillic fonts
- #21 Python 3 support
- #27 Code improvements
- #28 Human readable font list (-l)

[Unreleased]: https://github.com/Ace1928/figlet_forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ace1928/figlet_forge/releases/tag/v0.1.0
[0.0.1]: https://github.com/Ace1928/figlet_forge/releases/tag/v0.0.1
