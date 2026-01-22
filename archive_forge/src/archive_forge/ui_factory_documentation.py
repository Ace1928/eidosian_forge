import copy
Create a `base_ui.BaseUI` subtype.

  This factory method attempts to fallback to other available ui_types on
  ImportError.

  Args:
    ui_type: (`str`) requested UI type. Currently supported:
      ( readline)
    on_ui_exit: (`Callable`) the callback to be called when the UI exits.
    available_ui_types: (`None` or `list` of `str`) Manually-set available
      ui_types.
    config: An instance of `cli_config.CLIConfig()` carrying user-facing
      configurations.

  Returns:
    A `base_ui.BaseUI` subtype object.

  Raises:
    ValueError: on invalid ui_type or on exhausting or fallback ui_types.
  