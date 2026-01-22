import pytest
from spacy.tokens import Doc
from ...util import apply_transition_sequence
Test Issue #309: SBD fails on empty string