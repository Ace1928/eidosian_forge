class ASCIIFormatter:
    """ 
    takes text, adds seperators and decorating boxes with various embellishments, adds emoticons for key words and adds in fancy symbols for other key words in the input text, wraps up the date and time in a fancy formatted box and then puts all of these ASCII Graphical elements together in a structured manner and is able to handle essentially any kind of text input, be it from file, from lists of files, from lists of text, dictionaries, nested dictionaries etc.
    It recursively goes through more complex objects to break them up into component parts and utilises ASCII pipes and arrows to show the nesting structure correctly and fully in the output it formats and prepares and returns.
    """

    def __init__(
        self,
        input: Optional[Union[file, str, list[str], dict[str], dict[str, str]],dict[dict[dict[dict[str,str]]]]] = None,
    )
        """
    Initializes the AdvancedASCIIFormatter class with the default ASCII art components.
        """
        self.input = input
        self.NESTED_ENTRY_ARROW: str = "└──"  # For Nested Entry Arrow
        self.NESTED_ENTRY_PIPE: str = "│"  # For Nested Entry Pipe
        self.NESTED_ENTRY_SPACE: str = " "  # For Nested Entry Space
        self.NESTED_ENTRY_INDENT: str = "   "  # For Nested Entry Indent
        self.NESTED_ENTRY_HANGING: str = "├──"  # For Nested Entry Hanging
        self.NESTED_ENTRY_OPEN: str = "┌"  # For Nested Entry Open
        self.NESTED_ENTRY_CLOSE: str = "└"  # For Nested Entry Close
        self.QUESTION: str = "❓"   # For Marking anything that needs attention
        self.EXCLAMATION: str = "❗"   # For Marking anything urgent
        self.CHECK: str = "✅"   # For Marking anything that is good or correct
        self.CROSS: str = "❌"   # For Marking anything that is bad or incorrect
        self.NEWLINE: str = "\n"  # For Newline
        self.SEPARATOR: str = "-" * 80
        self.TOP_LEFT_CORNER: str = "┌"  # For Message Box Outline
        self.TOP_RIGHT_CORNER: str = "┐"  # For Message Box Outline
        self.BOTTOM_LEFT_CORNER: str = "└"  # For Message Box Outline
        self.BOTTOM_RIGHT_CORNER: str = "┘"  # For Message Box Outline
        self.HORIZONTAL_LINE: str = "─"  # For Message Box Outline
        self.VERTICAL_LINE: str = "│"  # For Message Box Outline
        self.HORIZONAL_DIVIDER_LEFT: str = (
            "├"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.HORIZONAL_DIVIDER_RIGHT: str = (
            "┤"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.HORIZONTAL_DIVIDER_MIDDLE: str = (
            "┼"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.VERTICAL_DIVIDER: str = (
            "┼"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_TOP_LEFT: str = (
            "╔"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_BOTTOM_LEFT: str = (
            "╚"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_TOP_RIGHT: str = (
            "╗"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_BOTTOM_RIGHT: str = (
            "╝"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_LEFT: str = (
            "╠"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_RIGHT: str = (
            "╣"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_MIDDLE: str = (
            "╬"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_VERTICAL: str = (
            "║"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_HORIZONTAL: str = (
            "═"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        # Level symbols for different log levels
        self.LEVEL_SYMBOLS: = {
            .NOTSET: "🤷",
            .DEBUG: "🐞",
            .INFO: "ℹ️",
            .WARNING: "⚠️",
            .ERROR: "❌",
            .CRITICAL: "🚨",
        }
        # Exception Symbols for different exception types
        self.EXCEPTION_SYMBOLS: = {
            ValueError: "#❌#",  # Value Error
            TypeError: "T❌T",  # Type Error
            KeyError: "K❌K",  # Key Error
            IndexError: "I❌I",  # Index Error
            AttributeError: "A❌A",  # Attribute Error
            Exception: "E❌E",  # General Exception
        }

    def __call__(
        self, log: LoggerConfig.LogEntryDictType
    )
        """
        """
        # Initialize a dictionary to hold the formatted input components
        formatted_input = {}

        # Format a header for marking clearly the start of the input
        

        # Format the timestamp component of the input
        

        # Format the level component of the input
        

        # Format the message component of the input
        
        # Format the module component of the input
        
        # Format the function component of the input
        
        # Format the line component of the input
        
        # Format the exception component of the input
        
        # Add In the Authorship Details and a Footer Section to act as a clear separator and identifier for the input end
        
        # Combine the exception and authorship components
        
        # Return the processed input dictionary with enhanced formatting
        

    def format(
        self, log: LoggerConfig.LogEntryDictType
    ) -> LoggerConfig.LogEntryDictType:
        """ """
        # If the input is a string, process it as a message and split it into parts
        if isinstance(log, str):
            # Split the message into parts and process each part
            parts = self.split_message(log)
            # Initialize a dictionary to hold the formatted input components
            dict: LoggerConfig.LogEntryDictType = {}
            # Process each part of the message
            for part in parts:
                # Process the part and update the dictionary with the formatted components
                dict.update(self(part))
            # Return the dictionary containing the formatted input components
            return dict
        # If the input is a dictionary, process it as a input
        elif isinstance(log, dict):
            # Process the input and return the formatted dictionary
            return self(log)
        # If the input is neither a string nor a dictionary, raise a TypeError
        else:
            raise TypeError("input must be a string or a dictionary")


class ColoredFormatter:
    """ """

    def __init__(self, log: LoggerConfig.LogEntryDictType) -> None:
        """ """
        self.COLORS: Dict[LoggerConfig.LogLevel, str] = {
            LoggerConfig.LogLevel.DEBUG: "\033[94m",
            LoggerConfig.LogLevel.INFO: "\033[92m",
            LoggerConfig.LogLevel.WARNING: "\033[93m",
            LoggerConfig.LogLevel.ERROR: "\033[91m",
            LoggerConfig.LogLevel.CRITICAL: "\033[95m",
        }
        self.RESET_COLOR: str = "\033[0m"
        self.log = log

    def __call__(
        self, log_entry: LoggerConfig.LogEntryDictType
    ) -> LoggerConfig.LogEntryDictType:
        """
        Formats a input with color based on its log level.

        Args:
            log_entry (Union[LogEntry, CollapsibleLogEntry]): The input to format.

        Returns:
            str: The formatted input with color.
        """
        color: str = self.COLORS.get(log.level, "")
        return super().__call__(
            f"{color}{log.timestamp} | {log.level.name.upper()} | {log.message}{self.RESET_COLOR}"
        )
