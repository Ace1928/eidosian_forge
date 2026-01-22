import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
Check names if they are allowed.
		If ``allow_magic_methods is True`` names in `__allowed_magic_methods__`
		are additionally allowed although their names start with `_`.
		