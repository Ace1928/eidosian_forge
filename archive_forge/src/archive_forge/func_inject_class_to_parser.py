from __future__ import annotations
import argparse
import os
import typing as t
def inject_class_to_parser(self, cls: t.Any) -> None:
    """Add dummy arguments to our ArgumentParser for the traits of this class

        The argparse-based loader currently does not actually add any class traits to
        the constructed ArgumentParser, only the flags & aliaes. In order to work nicely
        with argcomplete's completers functionality, this method adds dummy arguments
        of the form --Class.trait to the ArgumentParser instance.

        This method should be called selectively to reduce runtime overhead and to avoid
        spamming options across all of Application.classes.
        """
    try:
        for traitname, trait in cls.class_traits(config=True).items():
            completer = trait.metadata.get('argcompleter') or getattr(trait, 'argcompleter', None)
            multiplicity = trait.metadata.get('multiplicity')
            self._parser.add_argument(f'--{cls.__name__}.{traitname}', type=str, help=trait.help, nargs=multiplicity).completer = completer
    except AttributeError:
        pass