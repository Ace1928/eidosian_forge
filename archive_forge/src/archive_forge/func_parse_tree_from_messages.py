from __future__ import annotations
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, time, timedelta
from typing import (
from typing_extensions import TypeAlias
from streamlit import type_util, util
from streamlit.elements.heading import HeadingProtoTag
from streamlit.elements.widgets.select_slider import SelectSliderSerde
from streamlit.elements.widgets.slider import (
from streamlit.elements.widgets.time_widgets import (
from streamlit.proto.Alert_pb2 import Alert as AlertProto
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.proto.Button_pb2 import Button as ButtonProto
from streamlit.proto.ChatInput_pb2 import ChatInput as ChatInputProto
from streamlit.proto.Checkbox_pb2 import Checkbox as CheckboxProto
from streamlit.proto.Code_pb2 import Code as CodeProto
from streamlit.proto.ColorPicker_pb2 import ColorPicker as ColorPickerProto
from streamlit.proto.DateInput_pb2 import DateInput as DateInputProto
from streamlit.proto.Element_pb2 import Element as ElementProto
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.Heading_pb2 import Heading as HeadingProto
from streamlit.proto.Json_pb2 import Json as JsonProto
from streamlit.proto.Markdown_pb2 import Markdown as MarkdownProto
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.proto.MultiSelect_pb2 import MultiSelect as MultiSelectProto
from streamlit.proto.NumberInput_pb2 import NumberInput as NumberInputProto
from streamlit.proto.Radio_pb2 import Radio as RadioProto
from streamlit.proto.Selectbox_pb2 import Selectbox as SelectboxProto
from streamlit.proto.Slider_pb2 import Slider as SliderProto
from streamlit.proto.Text_pb2 import Text as TextProto
from streamlit.proto.TextArea_pb2 import TextArea as TextAreaProto
from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto
from streamlit.proto.TimeInput_pb2 import TimeInput as TimeInputProto
from streamlit.proto.Toast_pb2 import Toast as ToastProto
from streamlit.proto.WidgetStates_pb2 import WidgetState, WidgetStates
from streamlit.runtime.state.common import TESTING_KEY, user_key_from_widget_id
from streamlit.runtime.state.safe_session_state import SafeSessionState
def parse_tree_from_messages(messages: list[ForwardMsg]) -> ElementTree:
    """Transform a list of `ForwardMsg` into a tree matching the implicit
    tree structure of blocks and elements in a streamlit app.

    Returns the root of the tree, which acts as the entrypoint for the query
    and interaction API.
    """
    root = ElementTree()
    root.children = {0: SpecialBlock(type='main', root=root, proto=None), 1: SpecialBlock(type='sidebar', root=root, proto=None), 2: SpecialBlock(type='event', root=root, proto=None)}
    for msg in messages:
        if not msg.HasField('delta'):
            continue
        delta_path = msg.metadata.delta_path
        delta = msg.delta
        if delta.WhichOneof('type') == 'new_element':
            elt = delta.new_element
            ty = elt.WhichOneof('type')
            new_node: Node
            if ty == 'alert':
                format = elt.alert.format
                if format == AlertProto.Format.ERROR:
                    new_node = Error(elt.alert, root=root)
                elif format == AlertProto.Format.INFO:
                    new_node = Info(elt.alert, root=root)
                elif format == AlertProto.Format.SUCCESS:
                    new_node = Success(elt.alert, root=root)
                elif format == AlertProto.Format.WARNING:
                    new_node = Warning(elt.alert, root=root)
                else:
                    raise ValueError(f'Unknown alert type with format {elt.alert.format}')
            elif ty == 'arrow_data_frame':
                new_node = Dataframe(elt.arrow_data_frame, root=root)
            elif ty == 'arrow_table':
                new_node = Table(elt.arrow_table, root=root)
            elif ty == 'button':
                new_node = Button(elt.button, root=root)
            elif ty == 'chat_input':
                new_node = ChatInput(elt.chat_input, root=root)
            elif ty == 'checkbox':
                style = elt.checkbox.type
                if style == CheckboxProto.StyleType.TOGGLE:
                    new_node = Toggle(elt.checkbox, root=root)
                else:
                    new_node = Checkbox(elt.checkbox, root=root)
            elif ty == 'code':
                new_node = Code(elt.code, root=root)
            elif ty == 'color_picker':
                new_node = ColorPicker(elt.color_picker, root=root)
            elif ty == 'date_input':
                new_node = DateInput(elt.date_input, root=root)
            elif ty == 'exception':
                new_node = Exception(elt.exception, root=root)
            elif ty == 'heading':
                if elt.heading.tag == HeadingProtoTag.TITLE_TAG.value:
                    new_node = Title(elt.heading, root=root)
                elif elt.heading.tag == HeadingProtoTag.HEADER_TAG.value:
                    new_node = Header(elt.heading, root=root)
                elif elt.heading.tag == HeadingProtoTag.SUBHEADER_TAG.value:
                    new_node = Subheader(elt.heading, root=root)
                else:
                    raise ValueError(f'Unknown heading type with tag {elt.heading.tag}')
            elif ty == 'json':
                new_node = Json(elt.json, root=root)
            elif ty == 'markdown':
                if elt.markdown.element_type == MarkdownProto.Type.NATIVE:
                    new_node = Markdown(elt.markdown, root=root)
                elif elt.markdown.element_type == MarkdownProto.Type.CAPTION:
                    new_node = Caption(elt.markdown, root=root)
                elif elt.markdown.element_type == MarkdownProto.Type.LATEX:
                    new_node = Latex(elt.markdown, root=root)
                elif elt.markdown.element_type == MarkdownProto.Type.DIVIDER:
                    new_node = Divider(elt.markdown, root=root)
                else:
                    raise ValueError(f'Unknown markdown type {elt.markdown.element_type}')
            elif ty == 'metric':
                new_node = Metric(elt.metric, root=root)
            elif ty == 'multiselect':
                new_node = Multiselect(elt.multiselect, root=root)
            elif ty == 'number_input':
                new_node = NumberInput(elt.number_input, root=root)
            elif ty == 'radio':
                new_node = Radio(elt.radio, root=root)
            elif ty == 'selectbox':
                new_node = Selectbox(elt.selectbox, root=root)
            elif ty == 'slider':
                if elt.slider.type == SliderProto.Type.SLIDER:
                    new_node = Slider(elt.slider, root=root)
                elif elt.slider.type == SliderProto.Type.SELECT_SLIDER:
                    new_node = SelectSlider(elt.slider, root=root)
                else:
                    raise ValueError(f'Slider with unknown type {elt.slider}')
            elif ty == 'text':
                new_node = Text(elt.text, root=root)
            elif ty == 'text_area':
                new_node = TextArea(elt.text_area, root=root)
            elif ty == 'text_input':
                new_node = TextInput(elt.text_input, root=root)
            elif ty == 'time_input':
                new_node = TimeInput(elt.time_input, root=root)
            elif ty == 'toast':
                new_node = Toast(elt.toast, root=root)
            else:
                new_node = UnknownElement(elt, root=root)
        elif delta.WhichOneof('type') == 'add_block':
            block = delta.add_block
            bty = block.WhichOneof('type')
            if bty == 'chat_message':
                new_node = ChatMessage(block.chat_message, root=root)
            elif bty == 'column':
                new_node = Column(block.column, root=root)
            elif bty == 'expandable':
                if block.expandable.icon:
                    new_node = Status(block.expandable, root=root)
                else:
                    new_node = Expander(block.expandable, root=root)
            elif bty == 'tab':
                new_node = Tab(block.tab, root=root)
            else:
                new_node = Block(proto=block, root=root)
        else:
            continue
        current_node: Block = root
        for idx in delta_path[:-1]:
            children = current_node.children
            child = children.get(idx)
            if child is None:
                child = Block(proto=None, root=root)
                children[idx] = child
            assert isinstance(child, Block)
            current_node = child
        if isinstance(new_node, Block):
            placeholder_block = current_node.children.get(delta_path[-1])
            if placeholder_block is not None:
                new_node.children = placeholder_block.children
        current_node.children[delta_path[-1]] = new_node
    return root