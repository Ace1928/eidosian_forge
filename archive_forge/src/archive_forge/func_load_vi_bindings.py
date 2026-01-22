from __future__ import unicode_literals
from prompt_toolkit.buffer import ClipboardData, indent, unindent, reshape_text
from prompt_toolkit.document import Document
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import Filter, Condition, HasArg, Always, IsReadOnly
from prompt_toolkit.filters.cli import ViNavigationMode, ViInsertMode, ViInsertMultipleMode, ViReplaceMode, ViSelectionMode, ViWaitingForTextObjectMode, ViDigraphMode, ViMode
from prompt_toolkit.key_binding.digraphs import DIGRAPHS
from prompt_toolkit.key_binding.vi_state import CharacterFind, InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from prompt_toolkit.selection import SelectionType, SelectionState, PasteMode
from .scroll import scroll_forward, scroll_backward, scroll_half_page_up, scroll_half_page_down, scroll_one_line_up, scroll_one_line_down, scroll_page_up, scroll_page_down
from .named_commands import get_by_name
from ..registry import Registry, ConditionalRegistry, BaseRegistry
import prompt_toolkit.filters as filters
from six.moves import range
import codecs
import six
import string
def load_vi_bindings(get_search_state=None):
    """
    Vi extensions.

    # Overview of Readline Vi commands:
    # http://www.catonmat.net/download/bash-vi-editing-mode-cheat-sheet.pdf

     :param get_search_state: None or a callable that takes a
        CommandLineInterface and returns a SearchState.
    """
    registry = ConditionalRegistry(Registry(), ViMode())
    handle = registry.add_binding
    if get_search_state is None:

        def get_search_state(cli):
            return cli.search_state
    navigation_mode = ViNavigationMode()
    insert_mode = ViInsertMode()
    insert_multiple_mode = ViInsertMultipleMode()
    replace_mode = ViReplaceMode()
    selection_mode = ViSelectionMode()
    operator_given = ViWaitingForTextObjectMode()
    digraph_mode = ViDigraphMode()
    vi_transform_functions = [(('g', '?'), Always(), lambda string: codecs.encode(string, 'rot_13')), (('g', 'u'), Always(), lambda string: string.lower()), (('g', 'U'), Always(), lambda string: string.upper()), (('g', '~'), Always(), lambda string: string.swapcase()), (('~',), Condition(lambda cli: cli.vi_state.tilde_operator), lambda string: string.swapcase())]
    handle(Keys.ControlV, filter=insert_mode)(get_by_name('quoted-insert'))

    @handle(Keys.Escape)
    def _(event):
        """
        Escape goes to vi navigation mode.
        """
        buffer = event.current_buffer
        vi_state = event.cli.vi_state
        if vi_state.input_mode in (InputMode.INSERT, InputMode.REPLACE):
            buffer.cursor_position += buffer.document.get_cursor_left_position()
        vi_state.reset(InputMode.NAVIGATION)
        if bool(buffer.selection_state):
            buffer.exit_selection()

    @handle('k', filter=selection_mode)
    def _(event):
        """
        Arrow up in selection mode.
        """
        event.current_buffer.cursor_up(count=event.arg)

    @handle('j', filter=selection_mode)
    def _(event):
        """
        Arrow down in selection mode.
        """
        event.current_buffer.cursor_down(count=event.arg)

    @handle(Keys.Up, filter=navigation_mode)
    @handle(Keys.ControlP, filter=navigation_mode)
    def _(event):
        """
        Arrow up and ControlP in navigation mode go up.
        """
        event.current_buffer.auto_up(count=event.arg)

    @handle('k', filter=navigation_mode)
    def _(event):
        """
        Go up, but if we enter a new history entry, move to the start of the
        line.
        """
        event.current_buffer.auto_up(count=event.arg, go_to_start_of_line_if_history_changes=True)

    @handle(Keys.Down, filter=navigation_mode)
    @handle(Keys.ControlN, filter=navigation_mode)
    def _(event):
        """
        Arrow down and Control-N in navigation mode.
        """
        event.current_buffer.auto_down(count=event.arg)

    @handle('j', filter=navigation_mode)
    def _(event):
        """
        Go down, but if we enter a new history entry, go to the start of the line.
        """
        event.current_buffer.auto_down(count=event.arg, go_to_start_of_line_if_history_changes=True)

    @handle(Keys.ControlH, filter=navigation_mode)
    @handle(Keys.Backspace, filter=navigation_mode)
    def _(event):
        """
        In navigation-mode, move cursor.
        """
        event.current_buffer.cursor_position += event.current_buffer.document.get_cursor_left_position(count=event.arg)

    @handle(Keys.ControlN, filter=insert_mode)
    def _(event):
        b = event.current_buffer
        if b.complete_state:
            b.complete_next()
        else:
            event.cli.start_completion(select_first=True)

    @handle(Keys.ControlP, filter=insert_mode)
    def _(event):
        """
        Control-P: To previous completion.
        """
        b = event.current_buffer
        if b.complete_state:
            b.complete_previous()
        else:
            event.cli.start_completion(select_last=True)

    @handle(Keys.ControlY, filter=insert_mode)
    def _(event):
        """
        Accept current completion.
        """
        event.current_buffer.complete_state = None

    @handle(Keys.ControlE, filter=insert_mode)
    def _(event):
        """
        Cancel completion. Go back to originally typed text.
        """
        event.current_buffer.cancel_completion()

    @handle(Keys.ControlJ, filter=navigation_mode)
    def _(event):
        """
        In navigation mode, pressing enter will always return the input.
        """
        b = event.current_buffer
        if b.accept_action.is_returnable:
            b.accept_action.validate_and_handle(event.cli, b)

    @handle(Keys.Insert, filter=navigation_mode)
    def _(event):
        """ Presing the Insert key. """
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle('a', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        event.current_buffer.cursor_position += event.current_buffer.document.get_cursor_right_position()
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle('A', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        event.current_buffer.cursor_position += event.current_buffer.document.get_end_of_line_position()
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle('C', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """
        # Change to end of line.
        # Same as 'c$' (which is implemented elsewhere.)
        """
        buffer = event.current_buffer
        deleted = buffer.delete(count=buffer.document.get_end_of_line_position())
        event.cli.clipboard.set_text(deleted)
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle('c', 'c', filter=navigation_mode & ~IsReadOnly())
    @handle('S', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """
        Change current line
        """
        buffer = event.current_buffer
        data = ClipboardData(buffer.document.current_line, SelectionType.LINES)
        event.cli.clipboard.set_data(data)
        buffer.cursor_position += buffer.document.get_start_of_line_position(after_whitespace=True)
        buffer.delete(count=buffer.document.get_end_of_line_position())
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle('D', filter=navigation_mode)
    def _(event):
        buffer = event.current_buffer
        deleted = buffer.delete(count=buffer.document.get_end_of_line_position())
        event.cli.clipboard.set_text(deleted)

    @handle('d', 'd', filter=navigation_mode)
    def _(event):
        """
        Delete line. (Or the following 'n' lines.)
        """
        buffer = event.current_buffer
        lines = buffer.document.lines
        before = '\n'.join(lines[:buffer.document.cursor_position_row])
        deleted = '\n'.join(lines[buffer.document.cursor_position_row:buffer.document.cursor_position_row + event.arg])
        after = '\n'.join(lines[buffer.document.cursor_position_row + event.arg:])
        if before and after:
            before = before + '\n'
        buffer.document = Document(text=before + after, cursor_position=len(before) + len(after) - len(after.lstrip(' ')))
        event.cli.clipboard.set_data(ClipboardData(deleted, SelectionType.LINES))

    @handle('x', filter=selection_mode)
    def _(event):
        """
        Cut selection.
        ('x' is not an operator.)
        """
        clipboard_data = event.current_buffer.cut_selection()
        event.cli.clipboard.set_data(clipboard_data)

    @handle('i', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle('I', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        event.cli.vi_state.input_mode = InputMode.INSERT
        event.current_buffer.cursor_position += event.current_buffer.document.get_start_of_line_position(after_whitespace=True)

    @Condition
    def in_block_selection(cli):
        buff = cli.current_buffer
        return buff.selection_state and buff.selection_state.type == SelectionType.BLOCK

    @handle('I', filter=in_block_selection & ~IsReadOnly())
    def go_to_block_selection(event, after=False):
        """ Insert in block selection mode. """
        buff = event.current_buffer
        positions = []
        if after:

            def get_pos(from_to):
                return from_to[1] + 1
        else:

            def get_pos(from_to):
                return from_to[0]
        for i, from_to in enumerate(buff.document.selection_ranges()):
            positions.append(get_pos(from_to))
            if i == 0:
                buff.cursor_position = get_pos(from_to)
        buff.multiple_cursor_positions = positions
        event.cli.vi_state.input_mode = InputMode.INSERT_MULTIPLE
        buff.exit_selection()

    @handle('A', filter=in_block_selection & ~IsReadOnly())
    def _(event):
        go_to_block_selection(event, after=True)

    @handle('J', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """ Join lines. """
        for i in range(event.arg):
            event.current_buffer.join_next_line()

    @handle('g', 'J', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """ Join lines without space. """
        for i in range(event.arg):
            event.current_buffer.join_next_line(separator='')

    @handle('J', filter=selection_mode & ~IsReadOnly())
    def _(event):
        """ Join selected lines. """
        event.current_buffer.join_selected_lines()

    @handle('g', 'J', filter=selection_mode & ~IsReadOnly())
    def _(event):
        """ Join selected lines without space. """
        event.current_buffer.join_selected_lines(separator='')

    @handle('p', filter=navigation_mode)
    def _(event):
        """
        Paste after
        """
        event.current_buffer.paste_clipboard_data(event.cli.clipboard.get_data(), count=event.arg, paste_mode=PasteMode.VI_AFTER)

    @handle('P', filter=navigation_mode)
    def _(event):
        """
        Paste before
        """
        event.current_buffer.paste_clipboard_data(event.cli.clipboard.get_data(), count=event.arg, paste_mode=PasteMode.VI_BEFORE)

    @handle('"', Keys.Any, 'p', filter=navigation_mode)
    def _(event):
        """ Paste from named register. """
        c = event.key_sequence[1].data
        if c in vi_register_names:
            data = event.cli.vi_state.named_registers.get(c)
            if data:
                event.current_buffer.paste_clipboard_data(data, count=event.arg, paste_mode=PasteMode.VI_AFTER)

    @handle('"', Keys.Any, 'P', filter=navigation_mode)
    def _(event):
        """ Paste (before) from named register. """
        c = event.key_sequence[1].data
        if c in vi_register_names:
            data = event.cli.vi_state.named_registers.get(c)
            if data:
                event.current_buffer.paste_clipboard_data(data, count=event.arg, paste_mode=PasteMode.VI_BEFORE)

    @handle('r', Keys.Any, filter=navigation_mode)
    def _(event):
        """
        Replace single character under cursor
        """
        event.current_buffer.insert_text(event.data * event.arg, overwrite=True)
        event.current_buffer.cursor_position -= 1

    @handle('R', filter=navigation_mode)
    def _(event):
        """
        Go to 'replace'-mode.
        """
        event.cli.vi_state.input_mode = InputMode.REPLACE

    @handle('s', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """
        Substitute with new text
        (Delete character(s) and go to insert mode.)
        """
        text = event.current_buffer.delete(count=event.arg)
        event.cli.clipboard.set_text(text)
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle('u', filter=navigation_mode, save_before=lambda e: False)
    def _(event):
        for i in range(event.arg):
            event.current_buffer.undo()

    @handle('V', filter=navigation_mode)
    def _(event):
        """
        Start lines selection.
        """
        event.current_buffer.start_selection(selection_type=SelectionType.LINES)

    @handle(Keys.ControlV, filter=navigation_mode)
    def _(event):
        """ Enter block selection mode. """
        event.current_buffer.start_selection(selection_type=SelectionType.BLOCK)

    @handle('V', filter=selection_mode)
    def _(event):
        """
        Exit line selection mode, or go from non line selection mode to line
        selection mode.
        """
        selection_state = event.current_buffer.selection_state
        if selection_state.type != SelectionType.LINES:
            selection_state.type = SelectionType.LINES
        else:
            event.current_buffer.exit_selection()

    @handle('v', filter=navigation_mode)
    def _(event):
        """ Enter character selection mode. """
        event.current_buffer.start_selection(selection_type=SelectionType.CHARACTERS)

    @handle('v', filter=selection_mode)
    def _(event):
        """
        Exit character selection mode, or go from non-character-selection mode
        to character selection mode.
        """
        selection_state = event.current_buffer.selection_state
        if selection_state.type != SelectionType.CHARACTERS:
            selection_state.type = SelectionType.CHARACTERS
        else:
            event.current_buffer.exit_selection()

    @handle(Keys.ControlV, filter=selection_mode)
    def _(event):
        """
        Exit block selection mode, or go from non block selection mode to block
        selection mode.
        """
        selection_state = event.current_buffer.selection_state
        if selection_state.type != SelectionType.BLOCK:
            selection_state.type = SelectionType.BLOCK
        else:
            event.current_buffer.exit_selection()

    @handle('a', 'w', filter=selection_mode)
    @handle('a', 'W', filter=selection_mode)
    def _(event):
        """
        Switch from visual linewise mode to visual characterwise mode.
        """
        buffer = event.current_buffer
        if buffer.selection_state and buffer.selection_state.type == SelectionType.LINES:
            buffer.selection_state.type = SelectionType.CHARACTERS

    @handle('x', filter=navigation_mode)
    def _(event):
        """
        Delete character.
        """
        text = event.current_buffer.delete(count=event.arg)
        event.cli.clipboard.set_text(text)

    @handle('X', filter=navigation_mode)
    def _(event):
        text = event.current_buffer.delete_before_cursor()
        event.cli.clipboard.set_text(text)

    @handle('y', 'y', filter=navigation_mode)
    @handle('Y', filter=navigation_mode)
    def _(event):
        """
        Yank the whole line.
        """
        text = '\n'.join(event.current_buffer.document.lines_from_current[:event.arg])
        event.cli.clipboard.set_data(ClipboardData(text, SelectionType.LINES))

    @handle('+', filter=navigation_mode)
    def _(event):
        """
        Move to first non whitespace of next line
        """
        buffer = event.current_buffer
        buffer.cursor_position += buffer.document.get_cursor_down_position(count=event.arg)
        buffer.cursor_position += buffer.document.get_start_of_line_position(after_whitespace=True)

    @handle('-', filter=navigation_mode)
    def _(event):
        """
        Move to first non whitespace of previous line
        """
        buffer = event.current_buffer
        buffer.cursor_position += buffer.document.get_cursor_up_position(count=event.arg)
        buffer.cursor_position += buffer.document.get_start_of_line_position(after_whitespace=True)

    @handle('>', '>', filter=navigation_mode)
    def _(event):
        """
        Indent lines.
        """
        buffer = event.current_buffer
        current_row = buffer.document.cursor_position_row
        indent(buffer, current_row, current_row + event.arg)

    @handle('<', '<', filter=navigation_mode)
    def _(event):
        """
        Unindent lines.
        """
        current_row = event.current_buffer.document.cursor_position_row
        unindent(event.current_buffer, current_row, current_row + event.arg)

    @handle('O', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """
        Open line above and enter insertion mode
        """
        event.current_buffer.insert_line_above(copy_margin=not event.cli.in_paste_mode)
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle('o', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """
        Open line below and enter insertion mode
        """
        event.current_buffer.insert_line_below(copy_margin=not event.cli.in_paste_mode)
        event.cli.vi_state.input_mode = InputMode.INSERT

    @handle('~', filter=navigation_mode)
    def _(event):
        """
        Reverse case of current character and move cursor forward.
        """
        buffer = event.current_buffer
        c = buffer.document.current_char
        if c is not None and c != '\n':
            buffer.insert_text(c.swapcase(), overwrite=True)

    @handle('g', 'u', 'u', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """ Lowercase current line. """
        buff = event.current_buffer
        buff.transform_current_line(lambda s: s.lower())

    @handle('g', 'U', 'U', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """ Uppercase current line. """
        buff = event.current_buffer
        buff.transform_current_line(lambda s: s.upper())

    @handle('g', '~', '~', filter=navigation_mode & ~IsReadOnly())
    def _(event):
        """ Swap case of the current line. """
        buff = event.current_buffer
        buff.transform_current_line(lambda s: s.swapcase())

    @handle('#', filter=navigation_mode)
    def _(event):
        """
        Go to previous occurence of this word.
        """
        b = event.cli.current_buffer
        search_state = get_search_state(event.cli)
        search_state.text = b.document.get_word_under_cursor()
        search_state.direction = IncrementalSearchDirection.BACKWARD
        b.apply_search(search_state, count=event.arg, include_current_position=False)

    @handle('*', filter=navigation_mode)
    def _(event):
        """
        Go to next occurence of this word.
        """
        b = event.cli.current_buffer
        search_state = get_search_state(event.cli)
        search_state.text = b.document.get_word_under_cursor()
        search_state.direction = IncrementalSearchDirection.FORWARD
        b.apply_search(search_state, count=event.arg, include_current_position=False)

    @handle('(', filter=navigation_mode)
    def _(event):
        pass

    @handle(')', filter=navigation_mode)
    def _(event):
        pass
    operator = create_operator_decorator(registry)
    text_object = create_text_object_decorator(registry)

    @text_object(Keys.Any, filter=operator_given)
    def _(event):
        """
        Unknown key binding while waiting for a text object.
        """
        event.cli.output.bell()

    def create_delete_and_change_operators(delete_only, with_register=False):
        """
        Delete and change operators.

        :param delete_only: Create an operator that deletes, but doesn't go to insert mode.
        :param with_register: Copy the deleted text to this named register instead of the clipboard.
        """
        if with_register:
            handler_keys = ('"', Keys.Any, 'cd'[delete_only])
        else:
            handler_keys = 'cd'[delete_only]

        @operator(*handler_keys, filter=~IsReadOnly())
        def delete_or_change_operator(event, text_object):
            clipboard_data = None
            buff = event.current_buffer
            if text_object:
                new_document, clipboard_data = text_object.cut(buff)
                buff.document = new_document
            if clipboard_data and clipboard_data.text:
                if with_register:
                    reg_name = event.key_sequence[1].data
                    if reg_name in vi_register_names:
                        event.cli.vi_state.named_registers[reg_name] = clipboard_data
                else:
                    event.cli.clipboard.set_data(clipboard_data)
            if not delete_only:
                event.cli.vi_state.input_mode = InputMode.INSERT
    create_delete_and_change_operators(False, False)
    create_delete_and_change_operators(False, True)
    create_delete_and_change_operators(True, False)
    create_delete_and_change_operators(True, True)

    def create_transform_handler(filter, transform_func, *a):

        @operator(*a, filter=filter & ~IsReadOnly())
        def _(event, text_object):
            """
            Apply transformation (uppercase, lowercase, rot13, swap case).
            """
            buff = event.current_buffer
            start, end = text_object.operator_range(buff.document)
            if start < end:
                buff.transform_region(buff.cursor_position + start, buff.cursor_position + end, transform_func)
                buff.cursor_position += text_object.end or text_object.start
    for k, f, func in vi_transform_functions:
        create_transform_handler(f, func, *k)

    @operator('y')
    def yank_handler(event, text_object):
        """
        Yank operator. (Copy text.)
        """
        _, clipboard_data = text_object.cut(event.current_buffer)
        if clipboard_data.text:
            event.cli.clipboard.set_data(clipboard_data)

    @operator('"', Keys.Any, 'y')
    def _(event, text_object):
        """ Yank selection to named register. """
        c = event.key_sequence[1].data
        if c in vi_register_names:
            _, clipboard_data = text_object.cut(event.current_buffer)
            event.cli.vi_state.named_registers[c] = clipboard_data

    @operator('>')
    def _(event, text_object):
        """
        Indent.
        """
        buff = event.current_buffer
        from_, to = text_object.get_line_numbers(buff)
        indent(buff, from_, to + 1, count=event.arg)

    @operator('<')
    def _(event, text_object):
        """
        Unindent.
        """
        buff = event.current_buffer
        from_, to = text_object.get_line_numbers(buff)
        unindent(buff, from_, to + 1, count=event.arg)

    @operator('g', 'q')
    def _(event, text_object):
        """
        Reshape text.
        """
        buff = event.current_buffer
        from_, to = text_object.get_line_numbers(buff)
        reshape_text(buff, from_, to)

    @text_object('b')
    def _(event):
        """ Move one word or token left. """
        return TextObject(event.current_buffer.document.find_start_of_previous_word(count=event.arg) or 0)

    @text_object('B')
    def _(event):
        """ Move one non-blank word left """
        return TextObject(event.current_buffer.document.find_start_of_previous_word(count=event.arg, WORD=True) or 0)

    @text_object('$')
    def key_dollar(event):
        """ 'c$', 'd$' and '$':  Delete/change/move until end of line. """
        return TextObject(event.current_buffer.document.get_end_of_line_position())

    @text_object('w')
    def _(event):
        """ 'word' forward. 'cw', 'dw', 'w': Delete/change/move one word.  """
        return TextObject(event.current_buffer.document.find_next_word_beginning(count=event.arg) or event.current_buffer.document.get_end_of_document_position())

    @text_object('W')
    def _(event):
        """ 'WORD' forward. 'cW', 'dW', 'W': Delete/change/move one WORD.  """
        return TextObject(event.current_buffer.document.find_next_word_beginning(count=event.arg, WORD=True) or event.current_buffer.document.get_end_of_document_position())

    @text_object('e')
    def _(event):
        """ End of 'word': 'ce', 'de', 'e' """
        end = event.current_buffer.document.find_next_word_ending(count=event.arg)
        return TextObject(end - 1 if end else 0, type=TextObjectType.INCLUSIVE)

    @text_object('E')
    def _(event):
        """ End of 'WORD': 'cE', 'dE', 'E' """
        end = event.current_buffer.document.find_next_word_ending(count=event.arg, WORD=True)
        return TextObject(end - 1 if end else 0, type=TextObjectType.INCLUSIVE)

    @text_object('i', 'w', no_move_handler=True)
    def _(event):
        """ Inner 'word': ciw and diw """
        start, end = event.current_buffer.document.find_boundaries_of_current_word()
        return TextObject(start, end)

    @text_object('a', 'w', no_move_handler=True)
    def _(event):
        """ A 'word': caw and daw """
        start, end = event.current_buffer.document.find_boundaries_of_current_word(include_trailing_whitespace=True)
        return TextObject(start, end)

    @text_object('i', 'W', no_move_handler=True)
    def _(event):
        """ Inner 'WORD': ciW and diW """
        start, end = event.current_buffer.document.find_boundaries_of_current_word(WORD=True)
        return TextObject(start, end)

    @text_object('a', 'W', no_move_handler=True)
    def _(event):
        """ A 'WORD': caw and daw """
        start, end = event.current_buffer.document.find_boundaries_of_current_word(WORD=True, include_trailing_whitespace=True)
        return TextObject(start, end)

    @text_object('a', 'p', no_move_handler=True)
    def _(event):
        """
        Auto paragraph.
        """
        start = event.current_buffer.document.start_of_paragraph()
        end = event.current_buffer.document.end_of_paragraph(count=event.arg)
        return TextObject(start, end)

    @text_object('^')
    def key_circumflex(event):
        """ 'c^', 'd^' and '^': Soft start of line, after whitespace. """
        return TextObject(event.current_buffer.document.get_start_of_line_position(after_whitespace=True))

    @text_object('0')
    def key_zero(event):
        """
        'c0', 'd0': Hard start of line, before whitespace.
        (The move '0' key is implemented elsewhere, because a '0' could also change the `arg`.)
        """
        return TextObject(event.current_buffer.document.get_start_of_line_position(after_whitespace=False))

    def create_ci_ca_handles(ci_start, ci_end, inner, key=None):
        """
        Delete/Change string between this start and stop character. But keep these characters.
        This implements all the ci", ci<, ci{, ci(, di", di<, ca", ca<, ... combinations.
        """

        def handler(event):
            if ci_start == ci_end:
                start = event.current_buffer.document.find_backwards(ci_start, in_current_line=False)
                end = event.current_buffer.document.find(ci_end, in_current_line=False)
            else:
                start = event.current_buffer.document.find_enclosing_bracket_left(ci_start, ci_end)
                end = event.current_buffer.document.find_enclosing_bracket_right(ci_start, ci_end)
            if start is not None and end is not None:
                offset = 0 if inner else 1
                return TextObject(start + 1 - offset, end + offset)
            else:
                return TextObject(0)
        if key is None:
            text_object('ai'[inner], ci_start, no_move_handler=True)(handler)
            text_object('ai'[inner], ci_end, no_move_handler=True)(handler)
        else:
            text_object('ai'[inner], key, no_move_handler=True)(handler)
    for inner in (False, True):
        for ci_start, ci_end in [('"', '"'), ("'", "'"), ('`', '`'), ('[', ']'), ('<', '>'), ('{', '}'), ('(', ')')]:
            create_ci_ca_handles(ci_start, ci_end, inner)
        create_ci_ca_handles('(', ')', inner, 'b')
        create_ci_ca_handles('{', '}', inner, 'B')

    @text_object('{')
    def _(event):
        """
        Move to previous blank-line separated section.
        Implements '{', 'c{', 'd{', 'y{'
        """
        index = event.current_buffer.document.start_of_paragraph(count=event.arg, before=True)
        return TextObject(index)

    @text_object('}')
    def _(event):
        """
        Move to next blank-line separated section.
        Implements '}', 'c}', 'd}', 'y}'
        """
        index = event.current_buffer.document.end_of_paragraph(count=event.arg, after=True)
        return TextObject(index)

    @text_object('f', Keys.Any)
    def _(event):
        """
        Go to next occurance of character. Typing 'fx' will move the
        cursor to the next occurance of character. 'x'.
        """
        event.cli.vi_state.last_character_find = CharacterFind(event.data, False)
        match = event.current_buffer.document.find(event.data, in_current_line=True, count=event.arg)
        if match:
            return TextObject(match, type=TextObjectType.INCLUSIVE)
        else:
            return TextObject(0)

    @text_object('F', Keys.Any)
    def _(event):
        """
        Go to previous occurance of character. Typing 'Fx' will move the
        cursor to the previous occurance of character. 'x'.
        """
        event.cli.vi_state.last_character_find = CharacterFind(event.data, True)
        return TextObject(event.current_buffer.document.find_backwards(event.data, in_current_line=True, count=event.arg) or 0)

    @text_object('t', Keys.Any)
    def _(event):
        """
        Move right to the next occurance of c, then one char backward.
        """
        event.cli.vi_state.last_character_find = CharacterFind(event.data, False)
        match = event.current_buffer.document.find(event.data, in_current_line=True, count=event.arg)
        if match:
            return TextObject(match - 1, type=TextObjectType.INCLUSIVE)
        else:
            return TextObject(0)

    @text_object('T', Keys.Any)
    def _(event):
        """
        Move left to the previous occurance of c, then one char forward.
        """
        event.cli.vi_state.last_character_find = CharacterFind(event.data, True)
        match = event.current_buffer.document.find_backwards(event.data, in_current_line=True, count=event.arg)
        return TextObject(match + 1 if match else 0)

    def repeat(reverse):
        """
        Create ',' and ';' commands.
        """

        @text_object(',' if reverse else ';')
        def _(event):
            pos = 0
            vi_state = event.cli.vi_state
            type = TextObjectType.EXCLUSIVE
            if vi_state.last_character_find:
                char = vi_state.last_character_find.character
                backwards = vi_state.last_character_find.backwards
                if reverse:
                    backwards = not backwards
                if backwards:
                    pos = event.current_buffer.document.find_backwards(char, in_current_line=True, count=event.arg)
                else:
                    pos = event.current_buffer.document.find(char, in_current_line=True, count=event.arg)
                    type = TextObjectType.INCLUSIVE
            if pos:
                return TextObject(pos, type=type)
            else:
                return TextObject(0)
    repeat(True)
    repeat(False)

    @text_object('h')
    @text_object(Keys.Left)
    def _(event):
        """ Implements 'ch', 'dh', 'h': Cursor left. """
        return TextObject(event.current_buffer.document.get_cursor_left_position(count=event.arg))

    @text_object('j', no_move_handler=True, no_selection_handler=True)
    def _(event):
        """ Implements 'cj', 'dj', 'j', ... Cursor up. """
        return TextObject(event.current_buffer.document.get_cursor_down_position(count=event.arg), type=TextObjectType.LINEWISE)

    @text_object('k', no_move_handler=True, no_selection_handler=True)
    def _(event):
        """ Implements 'ck', 'dk', 'k', ... Cursor up. """
        return TextObject(event.current_buffer.document.get_cursor_up_position(count=event.arg), type=TextObjectType.LINEWISE)

    @text_object('l')
    @text_object(' ')
    @text_object(Keys.Right)
    def _(event):
        """ Implements 'cl', 'dl', 'l', 'c ', 'd ', ' '. Cursor right. """
        return TextObject(event.current_buffer.document.get_cursor_right_position(count=event.arg))

    @text_object('H')
    def _(event):
        """
        Moves to the start of the visible region. (Below the scroll offset.)
        Implements 'cH', 'dH', 'H'.
        """
        w = find_window_for_buffer_name(event.cli, event.cli.current_buffer_name)
        b = event.current_buffer
        if w and w.render_info:
            pos = b.document.translate_row_col_to_index(w.render_info.first_visible_line(after_scroll_offset=True), 0) - b.cursor_position
        else:
            pos = -len(b.document.text_before_cursor)
        return TextObject(pos, type=TextObjectType.LINEWISE)

    @text_object('M')
    def _(event):
        """
        Moves cursor to the vertical center of the visible region.
        Implements 'cM', 'dM', 'M'.
        """
        w = find_window_for_buffer_name(event.cli, event.cli.current_buffer_name)
        b = event.current_buffer
        if w and w.render_info:
            pos = b.document.translate_row_col_to_index(w.render_info.center_visible_line(), 0) - b.cursor_position
        else:
            pos = -len(b.document.text_before_cursor)
        return TextObject(pos, type=TextObjectType.LINEWISE)

    @text_object('L')
    def _(event):
        """
        Moves to the end of the visible region. (Above the scroll offset.)
        """
        w = find_window_for_buffer_name(event.cli, event.cli.current_buffer_name)
        b = event.current_buffer
        if w and w.render_info:
            pos = b.document.translate_row_col_to_index(w.render_info.last_visible_line(before_scroll_offset=True), 0) - b.cursor_position
        else:
            pos = len(b.document.text_after_cursor)
        return TextObject(pos, type=TextObjectType.LINEWISE)

    @text_object('n', no_move_handler=True)
    def _(event):
        """ Search next. """
        buff = event.current_buffer
        cursor_position = buff.get_search_position(get_search_state(event.cli), include_current_position=False, count=event.arg)
        return TextObject(cursor_position - buff.cursor_position)

    @handle('n', filter=navigation_mode)
    def _(event):
        """ Search next in navigation mode. (This goes through the history.) """
        event.current_buffer.apply_search(get_search_state(event.cli), include_current_position=False, count=event.arg)

    @text_object('N', no_move_handler=True)
    def _(event):
        """ Search previous. """
        buff = event.current_buffer
        cursor_position = buff.get_search_position(~get_search_state(event.cli), include_current_position=False, count=event.arg)
        return TextObject(cursor_position - buff.cursor_position)

    @handle('N', filter=navigation_mode)
    def _(event):
        """ Search previous in navigation mode. (This goes through the history.) """
        event.current_buffer.apply_search(~get_search_state(event.cli), include_current_position=False, count=event.arg)

    @handle('z', '+', filter=navigation_mode | selection_mode)
    @handle('z', 't', filter=navigation_mode | selection_mode)
    @handle('z', Keys.ControlJ, filter=navigation_mode | selection_mode)
    def _(event):
        """
        Scrolls the window to makes the current line the first line in the visible region.
        """
        w = find_window_for_buffer_name(event.cli, event.cli.current_buffer_name)
        b = event.cli.current_buffer
        w.vertical_scroll = b.document.cursor_position_row

    @handle('z', '-', filter=navigation_mode | selection_mode)
    @handle('z', 'b', filter=navigation_mode | selection_mode)
    def _(event):
        """
        Scrolls the window to makes the current line the last line in the visible region.
        """
        w = find_window_for_buffer_name(event.cli, event.cli.current_buffer_name)
        w.vertical_scroll = 0

    @handle('z', 'z', filter=navigation_mode | selection_mode)
    def _(event):
        """
        Center Window vertically around cursor.
        """
        w = find_window_for_buffer_name(event.cli, event.cli.current_buffer_name)
        b = event.cli.current_buffer
        if w and w.render_info:
            info = w.render_info
            scroll_height = info.window_height // 2
            y = max(0, b.document.cursor_position_row - 1)
            height = 0
            while y > 0:
                line_height = info.get_height_for_line(y)
                if height + line_height < scroll_height:
                    height += line_height
                    y -= 1
                else:
                    break
            w.vertical_scroll = y

    @text_object('%')
    def _(event):
        """
        Implements 'c%', 'd%', '%, 'y%' (Move to corresponding bracket.)
        If an 'arg' has been given, go this this % position in the file.
        """
        buffer = event.current_buffer
        if event._arg:
            if 0 < event.arg <= 100:
                absolute_index = buffer.document.translate_row_col_to_index(int((event.arg * buffer.document.line_count - 1) / 100), 0)
                return TextObject(absolute_index - buffer.document.cursor_position, type=TextObjectType.LINEWISE)
            else:
                return TextObject(0)
        else:
            match = buffer.document.find_matching_bracket_position()
            if match:
                return TextObject(match, type=TextObjectType.INCLUSIVE)
            else:
                return TextObject(0)

    @text_object('|')
    def _(event):
        return TextObject(event.current_buffer.document.get_column_cursor_position(event.arg - 1))

    @text_object('g', 'g')
    def _(event):
        """
        Implements 'gg', 'cgg', 'ygg'
        """
        d = event.current_buffer.document
        if event._arg:
            return TextObject(d.translate_row_col_to_index(event.arg - 1, 0) - d.cursor_position, type=TextObjectType.LINEWISE)
        else:
            return TextObject(d.get_start_of_document_position(), type=TextObjectType.LINEWISE)

    @text_object('g', '_')
    def _(event):
        """
        Go to last non-blank of line.
        'g_', 'cg_', 'yg_', etc..
        """
        return TextObject(event.current_buffer.document.last_non_blank_of_current_line_position(), type=TextObjectType.INCLUSIVE)

    @text_object('g', 'e')
    def _(event):
        """
        Go to last character of previous word.
        'ge', 'cge', 'yge', etc..
        """
        prev_end = event.current_buffer.document.find_previous_word_ending(count=event.arg)
        return TextObject(prev_end - 1 if prev_end is not None else 0, type=TextObjectType.INCLUSIVE)

    @text_object('g', 'E')
    def _(event):
        """
        Go to last character of previous WORD.
        'gE', 'cgE', 'ygE', etc..
        """
        prev_end = event.current_buffer.document.find_previous_word_ending(count=event.arg, WORD=True)
        return TextObject(prev_end - 1 if prev_end is not None else 0, type=TextObjectType.INCLUSIVE)

    @text_object('g', 'm')
    def _(event):
        """
        Like g0, but half a screenwidth to the right. (Or as much as possible.)
        """
        w = find_window_for_buffer_name(event.cli, event.cli.current_buffer_name)
        buff = event.current_buffer
        if w and w.render_info:
            width = w.render_info.window_width
            start = buff.document.get_start_of_line_position(after_whitespace=False)
            start += int(min(width / 2, len(buff.document.current_line)))
            return TextObject(start, type=TextObjectType.INCLUSIVE)
        return TextObject(0)

    @text_object('G')
    def _(event):
        """
        Go to the end of the document. (If no arg has been given.)
        """
        buf = event.current_buffer
        return TextObject(buf.document.translate_row_col_to_index(buf.document.line_count - 1, 0) - buf.cursor_position, type=TextObjectType.LINEWISE)

    @handle('G', filter=HasArg())
    def _(event):
        """
        If an argument is given, move to this line in the  history. (for
        example, 15G)
        """
        event.current_buffer.go_to_history(event.arg - 1)
    for n in '123456789':

        @handle(n, filter=navigation_mode | selection_mode | operator_given)
        def _(event):
            """
            Always handle numberics in navigation mode as arg.
            """
            event.append_to_arg_count(event.data)

    @handle('0', filter=(navigation_mode | selection_mode | operator_given) & HasArg())
    def _(event):
        """ Zero when an argument was already give. """
        event.append_to_arg_count(event.data)

    @handle(Keys.Any, filter=replace_mode)
    def _(event):
        """
        Insert data at cursor position.
        """
        event.current_buffer.insert_text(event.data, overwrite=True)

    @handle(Keys.Any, filter=insert_multiple_mode, save_before=lambda e: not e.is_repeat)
    def _(event):
        """
        Insert data at multiple cursor positions at once.
        (Usually a result of pressing 'I' or 'A' in block-selection mode.)
        """
        buff = event.current_buffer
        original_text = buff.text
        text = []
        p = 0
        for p2 in buff.multiple_cursor_positions:
            text.append(original_text[p:p2])
            text.append(event.data)
            p = p2
        text.append(original_text[p:])
        new_cursor_positions = [p + i + 1 for i, p in enumerate(buff.multiple_cursor_positions)]
        buff.text = ''.join(text)
        buff.multiple_cursor_positions = new_cursor_positions
        buff.cursor_position += 1

    @handle(Keys.Backspace, filter=insert_multiple_mode)
    def _(event):
        """ Backspace, using multiple cursors. """
        buff = event.current_buffer
        original_text = buff.text
        deleted_something = False
        text = []
        p = 0
        for p2 in buff.multiple_cursor_positions:
            if p2 > 0 and original_text[p2 - 1] != '\n':
                text.append(original_text[p:p2 - 1])
                deleted_something = True
            else:
                text.append(original_text[p:p2])
            p = p2
        text.append(original_text[p:])
        if deleted_something:
            lengths = [len(part) for part in text[:-1]]
            new_cursor_positions = list(accumulate(lengths))
            buff.text = ''.join(text)
            buff.multiple_cursor_positions = new_cursor_positions
            buff.cursor_position -= 1
        else:
            event.cli.output.bell()

    @handle(Keys.Delete, filter=insert_multiple_mode)
    def _(event):
        """ Delete, using multiple cursors. """
        buff = event.current_buffer
        original_text = buff.text
        deleted_something = False
        text = []
        new_cursor_positions = []
        p = 0
        for p2 in buff.multiple_cursor_positions:
            text.append(original_text[p:p2])
            if p2 >= len(original_text) or original_text[p2] == '\n':
                p = p2
            else:
                p = p2 + 1
                deleted_something = True
        text.append(original_text[p:])
        if deleted_something:
            lengths = [len(part) for part in text[:-1]]
            new_cursor_positions = list(accumulate(lengths))
            buff.text = ''.join(text)
            buff.multiple_cursor_positions = new_cursor_positions
        else:
            event.cli.output.bell()

    @handle(Keys.ControlX, Keys.ControlL, filter=insert_mode)
    def _(event):
        """
        Pressing the ControlX - ControlL sequence in Vi mode does line
        completion based on the other lines in the document and the history.
        """
        event.current_buffer.start_history_lines_completion()

    @handle(Keys.ControlX, Keys.ControlF, filter=insert_mode)
    def _(event):
        """
        Complete file names.
        """
        pass

    @handle(Keys.ControlK, filter=insert_mode | replace_mode)
    def _(event):
        """ Go into digraph mode. """
        event.cli.vi_state.waiting_for_digraph = True

    @Condition
    def digraph_symbol_1_given(cli):
        return cli.vi_state.digraph_symbol1 is not None

    @handle(Keys.Any, filter=digraph_mode & ~digraph_symbol_1_given)
    def _(event):
        event.cli.vi_state.digraph_symbol1 = event.data

    @handle(Keys.Any, filter=digraph_mode & digraph_symbol_1_given)
    def _(event):
        """ Insert digraph. """
        try:
            code = (event.cli.vi_state.digraph_symbol1, event.data)
            if code not in DIGRAPHS:
                code = code[::-1]
            symbol = DIGRAPHS[code]
        except KeyError:
            event.cli.output.bell()
        else:
            overwrite = event.cli.vi_state.input_mode == InputMode.REPLACE
            event.current_buffer.insert_text(six.unichr(symbol), overwrite=overwrite)
            event.cli.vi_state.waiting_for_digraph = False
        finally:
            event.cli.vi_state.waiting_for_digraph = False
            event.cli.vi_state.digraph_symbol1 = None
    return registry