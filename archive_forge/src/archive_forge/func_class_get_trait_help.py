from __future__ import annotations
import logging
import typing as t
from copy import deepcopy
from textwrap import dedent
from traitlets.traitlets import (
from traitlets.utils import warnings
from traitlets.utils.bunch import Bunch
from traitlets.utils.text import indent, wrap_paragraphs
from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key
@classmethod
def class_get_trait_help(cls, trait: TraitType[t.Any, t.Any], inst: HasTraits | None=None, helptext: str | None=None) -> str:
    """Get the helptext string for a single trait.

        :param inst:
            If given, its current trait values will be used in place of
            the class default.
        :param helptext:
            If not given, uses the `help` attribute of the current trait.
        """
    assert inst is None or isinstance(inst, cls)
    lines = []
    header = f'--{cls.__name__}.{trait.name}'
    if isinstance(trait, (Container, Dict)):
        multiplicity = trait.metadata.get('multiplicity', 'append')
        if isinstance(trait, Dict):
            sample_value = '<key-1>=<value-1>'
        else:
            sample_value = '<%s-item-1>' % trait.__class__.__name__.lower()
        if multiplicity == 'append':
            header = f'{header}={sample_value}...'
        else:
            header = f'{header} {sample_value}...'
    else:
        header = f'{header}=<{trait.__class__.__name__}>'
    lines.append(header)
    if helptext is None:
        helptext = trait.help
    if helptext != '':
        helptext = '\n'.join(wrap_paragraphs(helptext, 76))
        lines.append(indent(helptext))
    if 'Enum' in trait.__class__.__name__:
        lines.append(indent('Choices: %s' % trait.info()))
    if inst is not None:
        lines.append(indent(f'Current: {getattr(inst, trait.name or '')!r}'))
    else:
        try:
            dvr = trait.default_value_repr()
        except Exception:
            dvr = None
        if dvr is not None:
            if len(dvr) > 64:
                dvr = dvr[:61] + '...'
            lines.append(indent('Default: %s' % dvr))
    return '\n'.join(lines)