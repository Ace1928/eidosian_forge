import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
def configured_view(self, *, discard_comments_on_read=True, auto_map_initial_line_whitespace=True, auto_resolve_ambiguous_fields=True, preserve_field_comments_on_field_updates=True, auto_map_final_newline_in_multiline_values=True):
    """Provide a Dict[str, str]-like view of this paragraph with non-standard parameters

        This method returns a dict-like object representing this paragraph that is
        optionally configured differently from the default view.

            >>> example_deb822_paragraph = '''
            ... Package: foo
            ... # Field comment (because it becomes just before a field)
            ... Depends: libfoo,
            ... # Inline comment (associated with the next line)
            ...          libbar,
            ... '''
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> paragraph = next(iter(dfile))
            >>> # With the defaults, you only deal with the semantic values
            >>> # - no leading or trailing whitespace on the first part of the value
            >>> paragraph["Package"]
            'foo'
            >>> # - no inline comments in multiline values (but whitespace will be present
            >>> #   subsequent lines.)
            >>> print(paragraph["Depends"])
            libfoo,
                     libbar,
            >>> paragraph['Foo'] = 'bar'
            >>> paragraph.get('Foo')
            'bar'
            >>> paragraph.get('Unknown-Field') is None
            True
            >>> # But you get asymmetric behaviour with set vs. get
            >>> paragraph['Foo'] = ' bar\\n'
            >>> paragraph['Foo']
            'bar'
            >>> paragraph['Bar'] = '     bar\\n#Comment\\n another value\\n'
            >>> # Note that the whitespace on the first line has been normalized.
            >>> print("Bar: " + paragraph['Bar'])
            Bar: bar
             another value
            >>> # The comment is present (in case you where wondering)
            >>> print(paragraph.get_kvpair_element('Bar').convert_to_text(), end='')
            Bar:     bar
            #Comment
             another value
            >>> # On the other hand, you can choose to see the values as they are
            >>> # - We will just reset the paragraph as a "nothing up my sleeve"
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> paragraph = next(iter(dfile))
            >>> nonstd_dictview = paragraph.configured_view(
            ...     discard_comments_on_read=False,
            ...     auto_map_initial_line_whitespace=False,
            ...     # For paragraphs with duplicate fields, you can choose to get an error
            ...     # rather than the dict picking the first value available.
            ...     auto_resolve_ambiguous_fields=False,
            ...     auto_map_final_newline_in_multiline_values=False,
            ... )
            >>> # Because we have reset the state, Foo and Bar are no longer there.
            >>> 'Bar' not in paragraph and 'Foo' not in paragraph
            True
            >>> # We can now see the comments (discard_comments_on_read=False)
            >>> # (The leading whitespace in front of "libfoo" is due to
            >>> #  auto_map_initial_line_whitespace=False)
            >>> print(nonstd_dictview["Depends"], end='')
             libfoo,
            # Inline comment (associated with the next line)
                     libbar,
            >>> # And all the optional whitespace on the first value line
            >>> # (auto_map_initial_line_whitespace=False)
            >>> nonstd_dictview["Package"] == ' foo\\n'
            True
            >>> # ... which will give you symmetric behaviour with set vs. get
            >>> nonstd_dictview['Foo'] = '  bar \\n'
            >>> nonstd_dictview['Foo']
            '  bar \\n'
            >>> nonstd_dictview['Bar'] = '  bar \\n#Comment\\n another value\\n'
            >>> nonstd_dictview['Bar']
            '  bar \\n#Comment\\n another value\\n'
            >>> # But then you get no help either.
            >>> try:
            ...     nonstd_dictview["Baz"] = "foo"
            ... except ValueError:
            ...     print("Rejected")
            Rejected
            >>> # With auto_map_initial_line_whitespace=False, you have to include minimum a newline
            >>> nonstd_dictview["Baz"] = "foo\\n"
            >>> # The absence of leading whitespace gives you the terse variant at the expensive
            >>> # readability
            >>> paragraph.get_kvpair_element('Baz').convert_to_text()
            'Baz:foo\\n'
            >>> # But because they are views, changes performed via one view is visible in the other
            >>> paragraph['Foo']
            'bar'
            >>> # The views show the values according to their own rules. Therefore, there is an
            >>> # asymmetric between paragraph['Foo'] and nonstd_dictview['Foo']
            >>> # Nevertheless, you can read or write the fields via either - enabling you to use
            >>> # the view that best suit your use-case for the given field.
            >>> 'Baz' in paragraph and nonstd_dictview.get('Baz') is not None
            True
            >>> # Deletion via the view also works
            >>> del nonstd_dictview['Baz']
            >>> 'Baz' not in paragraph and nonstd_dictview.get('Baz') is None
            True


        :param discard_comments_on_read: When getting a field value from the dict,
          this parameter decides how in-line comments are handled.  When setting
          the value, inline comments are still allowed and will be retained.
          However, keep in mind that this option makes getter and setter assymetric
          as a "get" following a "set" with inline comments will omit the comments
          even if they are there (see the code example).
        :param auto_map_initial_line_whitespace: Special-case the first value line
          by trimming unnecessary whitespace leaving only the value. For single-line
          values, all space including newline is pruned. For multi-line values, the
          newline is preserved / needed to distinguish the first line from the
          following lines.  When setting a value, this option normalizes the
          whitespace of the initial line of the value field.
          When this option is set to True makes the dictionary behave more like the
          original Deb822 module.
        :param preserve_field_comments_on_field_updates: Whether to preserve the field
          comments when mutating the field.
        :param auto_resolve_ambiguous_fields: This parameter is only relevant for paragraphs
          that contain the same field multiple times (these are generally invalid).  If the
          caller requests an ambiguous field from an invalid paragraph via a plain field name,
          the return dict-like object will refuse to resolve the field (not knowing which
          version to pick).  This parameter (if set to True) instead changes the error into
          assuming the caller wants the *first* variant.
        :param auto_map_final_newline_in_multiline_values: This parameter controls whether
          a multiline field with have / need a trailing newline. If True, the trailing
          newline is hidden on get and automatically added in set (if missing).
          When this option is set to True makes the dictionary behave more like the
          original Deb822 module.
        """
    return Deb822DictishParagraphWrapper(self, discard_comments_on_read=discard_comments_on_read, auto_map_initial_line_whitespace=auto_map_initial_line_whitespace, auto_resolve_ambiguous_fields=auto_resolve_ambiguous_fields, preserve_field_comments_on_field_updates=preserve_field_comments_on_field_updates, auto_map_final_newline_in_multiline_values=auto_map_final_newline_in_multiline_values)