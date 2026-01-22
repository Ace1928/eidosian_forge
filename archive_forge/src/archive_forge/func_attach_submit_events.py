from __future__ import annotations
import inspect
import json
import os
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Literal
from gradio_client.documentation import document
from gradio import Examples, utils
from gradio.blocks import Blocks
from gradio.components import (
from gradio.data_classes import InterfaceTypes
from gradio.events import Dependency, Events, on
from gradio.exceptions import RenderError
from gradio.flagging import CSVLogger, FlaggingCallback, FlagMethod
from gradio.layouts import Accordion, Column, Row, Tab, Tabs
from gradio.pipelines import load_from_pipeline
from gradio.themes import ThemeClass as Theme
def attach_submit_events(self, _submit_btn: Button | None, _stop_btn: Button | None) -> Dependency:
    if self.live:
        if self.interface_type == InterfaceTypes.OUTPUT_ONLY:
            if _submit_btn is None:
                raise RenderError('Submit button not rendered')
            super().load(self.fn, None, self.output_components)
            return _submit_btn.click(self.fn, None, self.output_components, api_name=self.api_name, preprocess=not self.api_mode, postprocess=not self.api_mode, batch=self.batch, max_batch_size=self.max_batch_size)
        else:
            events: list[Callable] = []
            streaming_event = False
            for component in self.input_components:
                if component.has_event('stream') and component.streaming:
                    events.append(component.stream)
                    streaming_event = True
                elif component.has_event('change'):
                    events.append(component.change)
            return on(events, self.fn, self.input_components, self.output_components, api_name=self.api_name, preprocess=not self.api_mode, postprocess=not self.api_mode, show_progress='hidden' if streaming_event else 'full', trigger_mode='always_last')
    else:
        if _submit_btn is None:
            raise RenderError('Submit button not rendered')
        fn = self.fn
        extra_output = []
        triggers = [_submit_btn.click] + [component.submit for component in self.input_components if component.has_event(Events.submit)]
        if _stop_btn:
            extra_output = [_submit_btn, _stop_btn]

            async def cleanup():
                return [Button(visible=True), Button(visible=False)]
            predict_event = on(triggers, utils.async_lambda(lambda: (Button(visible=False), Button(visible=True))), inputs=None, outputs=[_submit_btn, _stop_btn], queue=False, show_api=False).then(self.fn, self.input_components, self.output_components, api_name=self.api_name, scroll_to_output=True, preprocess=not self.api_mode, postprocess=not self.api_mode, batch=self.batch, max_batch_size=self.max_batch_size, concurrency_limit=self.concurrency_limit)
            final_event = predict_event.then(cleanup, inputs=None, outputs=extra_output, queue=False, show_api=False)
            _stop_btn.click(cleanup, inputs=None, outputs=[_submit_btn, _stop_btn], cancels=predict_event, queue=False, show_api=False)
            return final_event
        else:
            return on(triggers, fn, self.input_components, self.output_components, api_name=self.api_name, scroll_to_output=True, preprocess=not self.api_mode, postprocess=not self.api_mode, batch=self.batch, max_batch_size=self.max_batch_size, concurrency_limit=self.concurrency_limit)