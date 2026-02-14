# universal_components.py
"""
universal_components.py

An extensive and enhanced Python module containing:
1. A fully configurable logging system with JSON formatting.
2. Custom JSON encoder/decoder with extended logic for Enums and data classes.
3. A robust and complete set of computational instructions (Turing-complete).
4. A collection of mathematical functions, categorized and validated.
5. A set of irreducible logic symbols/operators, validated for uniqueness.
6. English characters and phonemes with concurrency-safe modifications.
7. A comprehensive Validator class that checks each subsystem for accuracy.
8. A Main function providing real-time validation results and exit codes.
9. Additional expansions for concurrency, error handling, logging verbosity, 
   and overall code clarity.

NOTE: This file can be executed directly to run all validations.
"""

import os
import sys
import re
import json
import threading
import logging
import logging.handlers
from enum import Enum, auto
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any, Union


# =============================================================================
# 1. Enhanced Logging Configuration
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Custom JSON Formatter for structured logging with both terminal and file outputs.
    Includes real-time explicit feedback on each record, including module, function name,
    timestamp, and line info.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def configure_logging(
    log_level: str = "INFO",
    log_file: str = "universal_components.log",
    max_bytes: int = 10**6,  # 1MB
    backup_count: int = 5
) -> None:
    """
    Configure a root logger with both console (stream) and file handlers.
    Logs are output as structured JSON for easy parsing and readability.

    Args:
        log_level (str): The minimum log level to capture (e.g., DEBUG, INFO, ERROR). 
                         Fetched from environment if not provided.
        log_file (str):  The file path for the rotating log file.
        max_bytes (int): Maximum size in bytes before log rotation occurs.
        backup_count (int): Number of rotated log files to keep.
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Formatter for JSON structured logs
    json_formatter = JSONFormatter()

    # Console Handler (stream to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(json_formatter)

    # File Handler with log rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler.setFormatter(json_formatter)

    # Clear existing handlers (prevents duplicate logs if re-configured)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add the newly configured handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# Initialize logging based on environment variables or defaults.
LOG_LEVEL = os.getenv("UNIVERSAL_COMPONENTS_LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("UNIVERSAL_COMPONENTS_LOG_FILE", "universal_components.log")
configure_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)


# =============================================================================
# 2. Custom JSON Encoder and Decoder
# =============================================================================

class UniversalJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder that handles:
     - Enum objects: converts to their name string.
     - Dataclasses: converts to dict via asdict().
     - Extra data structures as needed.
    """

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        elif is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def universal_json_decoder(dct: Dict[str, Any]) -> Any:
    """
    Custom JSON Decoder to handle advanced transformations if needed.
    Currently passes all dictionaries through as-is.

    Args:
        dct (dict): A dictionary to be processed.

    Returns:
        dict: The (potentially transformed) dictionary.
    """
    # Extend this decoder with additional logic for specialized data conversions.
    return dct


# =============================================================================
# 3. Computational Instructions
# =============================================================================

class ComputationalInstruction(Enum):
    """
    Enum representing a comprehensive list of computational instructions
    for a Turing-complete instruction set. Each instruction is described
    alongside an InstructionDetail data class for deeper metadata.
    """

    # A. Data Movement Instructions
    LOAD = auto()          # Load data from memory into a register.
    STORE = auto()         # Store data from a register into memory.
    MOVE = auto()          # Transfer data from one register to another.
    COPY = auto()          # Duplicate data from one memory location to another.

    # B. Arithmetic Instructions
    ADD = auto()           # Add two numbers.
    SUB = auto()           # Subtract one number from another.
    MULT = auto()          # Multiply two numbers.
    DIV = auto()           # Divide one number by another.
    MOD = auto()           # Compute the modulus (remainder) of division.
    EXP = auto()           # Exponentiation (raise a number to a power).
    SQRT = auto()          # Calculate the square root of a number.

    # C. Logical Instructions
    AND = auto()           # Logical AND operation.
    OR = auto()            # Logical OR operation.
    NOT = auto()           # Logical NOT operation.
    XOR = auto()           # Logical exclusive OR operation.
    NAND = auto()          # Logical NAND operation.
    NOR = auto()           # Logical NOR operation.
    XNOR = auto()          # Logical XNOR operation.

    # D. Control Flow Instructions
    JMP = auto()           # Unconditional jump to a specified instruction.
    JZ = auto()            # Jump if zero flag is set.
    JNZ = auto()           # Jump if zero flag is not set.
    JG = auto()            # Jump if greater.
    JL = auto()            # Jump if less.
    JE = auto()            # Jump if equal.
    JNE = auto()           # Jump if not equal.
    CALL = auto()          # Call a subroutine or function.
    RET = auto()           # Return from a subroutine or function.
    NOP = auto()           # No operation; does nothing for one cycle.

    # E. Stack Operations
    PUSH = auto()          # Push a value onto the stack.
    POP = auto()           # Pop a value from the stack.

    # F. Input/Output Instructions
    PRINT = auto()         # Output data to the console or display.
    READ = auto()          # Read input from the user or an input device.

    # G. Memory Management Instructions
    ALLOC = auto()         # Allocate memory.
    FREE = auto()          # Free allocated memory.
    SET_INDEX = auto()     # Set the value at a specific index in an array.
    GET_INDEX = auto()     # Retrieve the value from a specific index in an array.
    CREATE_ARRAY = auto()  # Initialize a new array with a specified size.

    # H. Function and Exception Handling Instructions
    DEFINE_FUNCTION = auto()   # Define a new function.
    CALL_FUNCTION = auto()     # Call a defined function.
    RETURN_FUNCTION = auto()   # Return from a function.
    TRY = auto()               # Begin a try block for exception handling.
    CATCH = auto()             # Begin a catch block to handle exceptions.
    THROW = auto()             # Throw an exception.
    EXCEPT = auto()            # Define an exception handler.

    # I. Bitwise Operations
    SHIFT_LEFT = auto()     # Bitwise left shift.
    SHIFT_RIGHT = auto()    # Bitwise right shift.
    ROTATE_LEFT = auto()    # Bitwise rotate left.
    ROTATE_RIGHT = auto()   # Bitwise rotate right.

    # J. Comparison Instructions
    CMP = auto()            # Compare two values and set flags accordingly.

    # K. Miscellaneous Instructions
    HALT = auto()           # Stop the execution of the program.
    DEFINE_CONSTANT = auto()# Define a constant value.
    DEFINE_VARIABLE = auto()# Define a variable in memory.

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class InstructionDetail:
    """
    Dataclass for holding extensive detail about a specific ComputationalInstruction.
    """
    name: ComputationalInstruction
    description: str
    operands: List[str] = field(default_factory=list)
    example_usage: Optional[str] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'InstructionDetail':
        """
        Create an InstructionDetail instance from a dictionary representation.
        Ensures the 'name' field is translated back to the ComputationalInstruction Enum.
        """
        name = ComputationalInstruction[data['name']]
        return InstructionDetail(
            name=name,
            description=data['description'],
            operands=data.get('operands', []),
            example_usage=data.get('example_usage')
        )


class ComputationalInstructions:
    """
    Class encapsulating all computational instructions, storing them along with
    InstructionDetail objects for clarity and structured metadata.
    """
    instructions: Dict[ComputationalInstruction, InstructionDetail] = {
        # Data Movement Instructions
        ComputationalInstruction.LOAD: InstructionDetail(
            name=ComputationalInstruction.LOAD,
            description="Load data from memory into a register.",
            operands=["register", "memory_address"],
            example_usage="LOAD R1, 0x1000"
        ),
        ComputationalInstruction.STORE: InstructionDetail(
            name=ComputationalInstruction.STORE,
            description="Store data from a register into memory.",
            operands=["register", "memory_address"],
            example_usage="STORE R1, 0x1000"
        ),
        ComputationalInstruction.MOVE: InstructionDetail(
            name=ComputationalInstruction.MOVE,
            description="Transfer data from one register to another.",
            operands=["destination_register", "source_register"],
            example_usage="MOVE R2, R1"
        ),
        ComputationalInstruction.COPY: InstructionDetail(
            name=ComputationalInstruction.COPY,
            description="Duplicate data from one memory location to another.",
            operands=["destination_memory", "source_memory"],
            example_usage="COPY 0x1004, 0x1000"
        ),

        # Arithmetic Instructions
        ComputationalInstruction.ADD: InstructionDetail(
            name=ComputationalInstruction.ADD,
            description="Add two numbers.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="ADD R3, R1, R2"
        ),
        ComputationalInstruction.SUB: InstructionDetail(
            name=ComputationalInstruction.SUB,
            description="Subtract one number from another.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="SUB R3, R1, R2"
        ),
        ComputationalInstruction.MULT: InstructionDetail(
            name=ComputationalInstruction.MULT,
            description="Multiply two numbers.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="MULT R3, R1, R2"
        ),
        ComputationalInstruction.DIV: InstructionDetail(
            name=ComputationalInstruction.DIV,
            description="Divide one number by another.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="DIV R3, R1, R2"
        ),
        ComputationalInstruction.MOD: InstructionDetail(
            name=ComputationalInstruction.MOD,
            description="Compute the modulus (remainder) of division.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="MOD R3, R1, R2"
        ),
        ComputationalInstruction.EXP: InstructionDetail(
            name=ComputationalInstruction.EXP,
            description="Exponentiation (raise a number to a power).",
            operands=["destination_register", "base_register", "exponent_register"],
            example_usage="EXP R3, R1, R2"
        ),
        ComputationalInstruction.SQRT: InstructionDetail(
            name=ComputationalInstruction.SQRT,
            description="Calculate the square root of a number.",
            operands=["destination_register", "source_register"],
            example_usage="SQRT R3, R1"
        ),

        # Logical Instructions
        ComputationalInstruction.AND: InstructionDetail(
            name=ComputationalInstruction.AND,
            description="Logical AND operation.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="AND R3, R1, R2"
        ),
        ComputationalInstruction.OR: InstructionDetail(
            name=ComputationalInstruction.OR,
            description="Logical OR operation.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="OR R3, R1, R2"
        ),
        ComputationalInstruction.NOT: InstructionDetail(
            name=ComputationalInstruction.NOT,
            description="Logical NOT operation.",
            operands=["destination_register", "source_register"],
            example_usage="NOT R3, R1"
        ),
        ComputationalInstruction.XOR: InstructionDetail(
            name=ComputationalInstruction.XOR,
            description="Logical exclusive OR operation.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="XOR R3, R1, R2"
        ),
        ComputationalInstruction.NAND: InstructionDetail(
            name=ComputationalInstruction.NAND,
            description="Logical NAND operation.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="NAND R3, R1, R2"
        ),
        ComputationalInstruction.NOR: InstructionDetail(
            name=ComputationalInstruction.NOR,
            description="Logical NOR operation.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="NOR R3, R1, R2"
        ),
        ComputationalInstruction.XNOR: InstructionDetail(
            name=ComputationalInstruction.XNOR,
            description="Logical XNOR operation.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="XNOR R3, R1, R2"
        ),

        # Control Flow Instructions
        ComputationalInstruction.JMP: InstructionDetail(
            name=ComputationalInstruction.JMP,
            description="Unconditional jump to a specified instruction.",
            operands=["instruction_label"],
            example_usage="JMP LOOP_START"
        ),
        ComputationalInstruction.JZ: InstructionDetail(
            name=ComputationalInstruction.JZ,
            description="Jump if zero flag is set.",
            operands=["instruction_label"],
            example_usage="JZ ZERO_FLAG_SET"
        ),
        ComputationalInstruction.JNZ: InstructionDetail(
            name=ComputationalInstruction.JNZ,
            description="Jump if zero flag is not set.",
            operands=["instruction_label"],
            example_usage="JNZ ZERO_FLAG_NOT_SET"
        ),
        ComputationalInstruction.JG: InstructionDetail(
            name=ComputationalInstruction.JG,
            description="Jump if greater.",
            operands=["instruction_label"],
            example_usage="JG GREATER_THAN"
        ),
        ComputationalInstruction.JL: InstructionDetail(
            name=ComputationalInstruction.JL,
            description="Jump if less.",
            operands=["instruction_label"],
            example_usage="JL LESS_THAN"
        ),
        ComputationalInstruction.JE: InstructionDetail(
            name=ComputationalInstruction.JE,
            description="Jump if equal.",
            operands=["instruction_label"],
            example_usage="JE EQUAL_TO"
        ),
        ComputationalInstruction.JNE: InstructionDetail(
            name=ComputationalInstruction.JNE,
            description="Jump if not equal.",
            operands=["instruction_label"],
            example_usage="JNE NOT_EQUAL_TO"
        ),
        ComputationalInstruction.CALL: InstructionDetail(
            name=ComputationalInstruction.CALL,
            description="Call a subroutine or function.",
            operands=["function_label"],
            example_usage="CALL SUBROUTINE_FUNC"
        ),
        ComputationalInstruction.RET: InstructionDetail(
            name=ComputationalInstruction.RET,
            description="Return from a subroutine or function.",
            operands=[],
            example_usage="RET"
        ),
        ComputationalInstruction.NOP: InstructionDetail(
            name=ComputationalInstruction.NOP,
            description="No operation; does nothing for one cycle.",
            operands=[],
            example_usage="NOP"
        ),

        # Stack Operations
        ComputationalInstruction.PUSH: InstructionDetail(
            name=ComputationalInstruction.PUSH,
            description="Push a value onto the stack.",
            operands=["register_or_value"],
            example_usage="PUSH R1"
        ),
        ComputationalInstruction.POP: InstructionDetail(
            name=ComputationalInstruction.POP,
            description="Pop a value from the stack.",
            operands=["register"],
            example_usage="POP R1"
        ),

        # Input/Output Instructions
        ComputationalInstruction.PRINT: InstructionDetail(
            name=ComputationalInstruction.PRINT,
            description="Output data to the console or display.",
            operands=["register_or_value"],
            example_usage='PRINT "Hello, World!"'
        ),
        ComputationalInstruction.READ: InstructionDetail(
            name=ComputationalInstruction.READ,
            description="Read input from the user or an input device.",
            operands=["register"],
            example_usage="READ R1"
        ),

        # Memory Management Instructions
        ComputationalInstruction.ALLOC: InstructionDetail(
            name=ComputationalInstruction.ALLOC,
            description="Allocate memory.",
            operands=["size"],
            example_usage="ALLOC 256"
        ),
        ComputationalInstruction.FREE: InstructionDetail(
            name=ComputationalInstruction.FREE,
            description="Free allocated memory.",
            operands=["memory_address"],
            example_usage="FREE 0x1000"
        ),
        ComputationalInstruction.SET_INDEX: InstructionDetail(
            name=ComputationalInstruction.SET_INDEX,
            description="Set the value at a specific index in an array.",
            operands=["array_address", "index", "value"],
            example_usage="SET_INDEX 0x2000, 5, 10"
        ),
        ComputationalInstruction.GET_INDEX: InstructionDetail(
            name=ComputationalInstruction.GET_INDEX,
            description="Retrieve the value from a specific index in an array.",
            operands=["destination_register", "array_address", "index"],
            example_usage="GET_INDEX R1, 0x2000, 5"
        ),
        ComputationalInstruction.CREATE_ARRAY: InstructionDetail(
            name=ComputationalInstruction.CREATE_ARRAY,
            description="Initialize a new array with a specified size.",
            operands=["array_label", "size"],
            example_usage="CREATE_ARRAY myArray, 10"
        ),

        # Function and Exception Handling Instructions
        ComputationalInstruction.DEFINE_FUNCTION: InstructionDetail(
            name=ComputationalInstruction.DEFINE_FUNCTION,
            description="Define a new function.",
            operands=["function_label"],
            example_usage="DEFINE_FUNCTION myFunction"
        ),
        ComputationalInstruction.CALL_FUNCTION: InstructionDetail(
            name=ComputationalInstruction.CALL_FUNCTION,
            description="Call a defined function.",
            operands=["function_label"],
            example_usage="CALL_FUNCTION myFunction"
        ),
        ComputationalInstruction.RETURN_FUNCTION: InstructionDetail(
            name=ComputationalInstruction.RETURN_FUNCTION,
            description="Return from a function.",
            operands=[],
            example_usage="RETURN"
        ),
        ComputationalInstruction.TRY: InstructionDetail(
            name=ComputationalInstruction.TRY,
            description="Begin a try block for exception handling.",
            operands=[],
            example_usage="TRY"
        ),
        ComputationalInstruction.CATCH: InstructionDetail(
            name=ComputationalInstruction.CATCH,
            description="Begin a catch block to handle exceptions.",
            operands=[],
            example_usage="CATCH"
        ),
        ComputationalInstruction.THROW: InstructionDetail(
            name=ComputationalInstruction.THROW,
            description="Throw an exception.",
            operands=[],
            example_usage="THROW"
        ),
        ComputationalInstruction.EXCEPT: InstructionDetail(
            name=ComputationalInstruction.EXCEPT,
            description="Define an exception handler.",
            operands=["exception_type"],
            example_usage="EXCEPT NullReferenceException"
        ),

        # Bitwise Operations
        ComputationalInstruction.SHIFT_LEFT: InstructionDetail(
            name=ComputationalInstruction.SHIFT_LEFT,
            description="Bitwise left shift.",
            operands=["register", "shift_amount"],
            example_usage="SHIFT_LEFT R1, 2"
        ),
        ComputationalInstruction.SHIFT_RIGHT: InstructionDetail(
            name=ComputationalInstruction.SHIFT_RIGHT,
            description="Bitwise right shift.",
            operands=["register", "shift_amount"],
            example_usage="SHIFT_RIGHT R1, 2"
        ),
        ComputationalInstruction.ROTATE_LEFT: InstructionDetail(
            name=ComputationalInstruction.ROTATE_LEFT,
            description="Bitwise rotate left.",
            operands=["register", "rotate_amount"],
            example_usage="ROTATE_LEFT R1, 1"
        ),
        ComputationalInstruction.ROTATE_RIGHT: InstructionDetail(
            name=ComputationalInstruction.ROTATE_RIGHT,
            description="Bitwise rotate right.",
            operands=["register", "rotate_amount"],
            example_usage="ROTATE_RIGHT R1, 1"
        ),

        # Comparison Instructions
        ComputationalInstruction.CMP: InstructionDetail(
            name=ComputationalInstruction.CMP,
            description="Compare two values and set flags accordingly.",
            operands=["register1", "register2"],
            example_usage="CMP R1, R2"
        ),

        # Miscellaneous Instructions
        ComputationalInstruction.HALT: InstructionDetail(
            name=ComputationalInstruction.HALT,
            description="Stop the execution of the program.",
            operands=[],
            example_usage="HALT"
        ),
        ComputationalInstruction.DEFINE_CONSTANT: InstructionDetail(
            name=ComputationalInstruction.DEFINE_CONSTANT,
            description="Define a constant value.",
            operands=["constant_label", "value"],
            example_usage="DEFINE_CONSTANT PI, 3.14159"
        ),
        ComputationalInstruction.DEFINE_VARIABLE: InstructionDetail(
            name=ComputationalInstruction.DEFINE_VARIABLE,
            description="Define a variable in memory.",
            operands=["variable_label", "initial_value"],
            example_usage="DEFINE_VARIABLE counter, 0"
        ),
    }

    @staticmethod
    def get_instruction_detail(instruction: ComputationalInstruction) -> Optional[InstructionDetail]:
        """
        Retrieve the instruction detail object if present.

        Args:
            instruction (ComputationalInstruction): The instruction to retrieve.

        Returns:
            InstructionDetail or None
        """
        return ComputationalInstructions.instructions.get(instruction)

    @staticmethod
    def to_json() -> str:
        """
        Serialize the entire instructions dictionary to a JSON string.

        Returns:
            str: JSON string representing all instructions.
        """
        serializable_dict = {
            instr.name: asdict(detail) for instr, detail in ComputationalInstructions.instructions.items()
        }
        return json.dumps(serializable_dict, indent=4, cls=UniversalJSONEncoder)

    @staticmethod
    def from_json(json_str: str) -> 'ComputationalInstructions':
        """
        Deserialize the JSON string into the instructions dictionary.

        Args:
            json_str (str): JSON string of instructions.

        Returns:
            ComputationalInstructions: The class itself (not an instance).
        """
        data = json.loads(json_str, object_hook=universal_json_decoder)
        for instr_name, detail in data.items():
            try:
                instr_enum = ComputationalInstruction[instr_name]
                ComputationalInstructions.instructions[instr_enum] = InstructionDetail.from_dict(detail)
            except KeyError:
                logging.warning(f"Unknown instruction '{instr_name}' found in JSON and will be skipped.")
        return ComputationalInstructions

    @staticmethod
    def add_instruction(detail: InstructionDetail) -> None:
        """
        Safely add a new computational instruction if it does not already exist.
        """
        if detail.name in ComputationalInstructions.instructions:
            logging.error(f"Instruction '{detail.name}' already exists.")
            raise ValueError(f"Instruction '{detail.name}' already exists.")
        ComputationalInstructions.instructions[detail.name] = detail
        logging.info(f"Added instruction '{detail.name}' successfully.")

    @staticmethod
    def remove_instruction(instruction: ComputationalInstruction) -> None:
        """
        Remove an existing computational instruction.

        Raises:
            KeyError: If the instruction does not exist.
        """
        if instruction not in ComputationalInstructions.instructions:
            logging.error(f"Instruction '{instruction}' does not exist.")
            raise KeyError(f"Instruction '{instruction}' does not exist.")
        del ComputationalInstructions.instructions[instruction]
        logging.info(f"Removed instruction '{instruction}' successfully.")

    @staticmethod
    def modify_instruction(instruction: ComputationalInstruction, **kwargs) -> None:
        """
        Modify details of an existing computational instruction.

        Raises:
            KeyError: If the instruction does not exist.
        """
        if instruction not in ComputationalInstructions.instructions:
            logging.error(f"Instruction '{instruction}' does not exist.")
            raise KeyError(f"Instruction '{instruction}' does not exist.")
        current_detail = ComputationalInstructions.instructions[instruction]
        updated_detail = InstructionDetail(
            name=current_detail.name,
            description=kwargs.get('description', current_detail.description),
            operands=kwargs.get('operands', current_detail.operands),
            example_usage=kwargs.get('example_usage', current_detail.example_usage)
        )
        ComputationalInstructions.instructions[instruction] = updated_detail
        logging.info(f"Modified instruction '{instruction}' successfully.")

    @staticmethod
    def list_all_instructions() -> List[InstructionDetail]:
        """
        List all defined computational instructions.
        """
        return list(ComputationalInstructions.instructions.values())

    @staticmethod
    def search_instructions(keyword: str) -> List[InstructionDetail]:
        """
        Search instructions by keyword in either name or description.

        Args:
            keyword (str): A substring to look for.

        Returns:
            List[InstructionDetail]: Matching instructions.
        """
        keyword_lower = keyword.lower()
        return [
            detail for detail in ComputationalInstructions.instructions.values()
            if keyword_lower in detail.name.name.lower() or keyword_lower in detail.description.lower()
        ]


# =============================================================================
# 4. Mathematical Functions
# =============================================================================

class MathematicalFunctionCategory(Enum):
    """
    Enum representing categories of non-composite mathematical functions.
    """

    POLYNOMIAL = auto()
    EXPONENTIAL_LOGARITHMIC = auto()
    TRIGONOMETRIC = auto()
    INVERSE_TRIGONOMETRIC = auto()
    HYPERBOLIC = auto()
    INVERSE_HYPERBOLIC = auto()
    SPECIAL = auto()
    ABSOLUTE_PIECEWISE = auto()
    MISCELLANEOUS = auto()


@dataclass(frozen=True)
class MathematicalFunction:
    """
    Dataclass representing a single mathematical function with category, symbol,
    domain, range, properties, and related functions.
    """
    name: str
    symbol: str
    category: MathematicalFunctionCategory
    description: str = ""
    domain: Optional[str] = None
    range: Optional[str] = None
    properties: Optional[List[str]] = field(default_factory=list)
    related_functions: Optional[List[str]] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'MathematicalFunction':
        """
        Deserialize a dictionary into a MathematicalFunction, mapping string 'category' to an Enum member.
        """
        category = MathematicalFunctionCategory[data['category']]
        return MathematicalFunction(
            name=data['name'],
            symbol=data['symbol'],
            category=category,
            description=data.get('description', ""),
            domain=data.get('domain'),
            range=data.get('range'),
            properties=data.get('properties', []),
            related_functions=data.get('related_functions', [])
        )


class MathematicalFunctions:
    """
    Class encapsulating all non-composite mathematical functions.
    Organized by categories in a 'functions' dictionary.
    """

    functions: Dict[MathematicalFunctionCategory, List[MathematicalFunction]] = {
        MathematicalFunctionCategory.POLYNOMIAL: [
            MathematicalFunction(
                name="Constant Function",
                symbol="f(x) = c",
                category=MathematicalFunctionCategory.POLYNOMIAL,
                description="A function that always returns a constant value.",
                domain="All real numbers",
                range="c",
                properties=["Linear", "Degree 0"],
                related_functions=["Linear Function"]
            ),
            MathematicalFunction(
                name="Linear Function",
                symbol="f(x) = mx + b",
                category=MathematicalFunctionCategory.POLYNOMIAL,
                description="A first-degree polynomial function representing a straight line.",
                domain="All real numbers",
                range="All real numbers",
                properties=["Continuous", "One-to-one"],
                related_functions=["Quadratic Function"]
            ),
            MathematicalFunction(
                name="Quadratic Function",
                symbol="f(x) = ax² + bx + c",
                category=MathematicalFunctionCategory.POLYNOMIAL,
                description="A second-degree polynomial function representing a parabola.",
                domain="All real numbers",
                range="All real numbers",
                properties=["Symmetrical", "Can have maximum or minimum"],
                related_functions=["Cubic Function"]
            ),
            MathematicalFunction(
                name="Cubic Function",
                symbol="f(x) = ax³ + bx² + cx + d",
                category=MathematicalFunctionCategory.POLYNOMIAL,
                description="A third-degree polynomial function representing an S-shaped curve.",
                domain="All real numbers",
                range="All real numbers",
                properties=["Can have inflection point", "One-to-one"],
                related_functions=["Quartic Function"]
            ),
            MathematicalFunction(
                name="Quartic Function",
                symbol="f(x) = ax⁴ + bx³ + cx² + dx + e",
                category=MathematicalFunctionCategory.POLYNOMIAL,
                description="A fourth-degree polynomial function with up to four real roots.",
                domain="All real numbers",
                range="All real numbers",
                properties=["Can have multiple turning points"],
                related_functions=["Quintic Function"]
            ),
            MathematicalFunction(
                name="Quintic Function",
                symbol="f(x) = ax⁵ + bx⁴ + cx³ + dx² + ex + f",
                category=MathematicalFunctionCategory.POLYNOMIAL,
                description="A fifth-degree polynomial function with up to five real roots.",
                domain="All real numbers",
                range="All real numbers",
                properties=["More complex behavior", "Can have multiple inflection points"],
                related_functions=["General Polynomial"]
            ),
            MathematicalFunction(
                name="General Polynomial",
                symbol="f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀",
                category=MathematicalFunctionCategory.POLYNOMIAL,
                description="A polynomial function of degree n with coefficients aₙ to a₀.",
                domain="All real numbers",
                range="All real numbers",
                properties=["Depends on degree", "Can have multiple roots"],
                related_functions=["All Polynomial Functions"]
            ),
        ],
        MathematicalFunctionCategory.EXPONENTIAL_LOGARITHMIC: [
            MathematicalFunction(
                name="Exponential Function",
                symbol="f(x) = e^x",
                category=MathematicalFunctionCategory.EXPONENTIAL_LOGARITHMIC,
                description="A function representing exponential growth with base e.",
                domain="All real numbers",
                range="(0, ∞)",
                properties=["Continuous", "Always increasing"],
                related_functions=["Natural Logarithm"]
            ),
            MathematicalFunction(
                name="General Exponential Function",
                symbol="f(x) = a^x",
                category=MathematicalFunctionCategory.EXPONENTIAL_LOGARITHMIC,
                description="An exponential function with arbitrary positive base a.",
                domain="All real numbers",
                range="(0, ∞)",
                properties=["Continuous", "Always increasing if a > 1"],
                related_functions=["Inverse Exponential Function"]
            ),
            MathematicalFunction(
                name="Natural Logarithm",
                symbol="f(x) = ln(x)",
                category=MathematicalFunctionCategory.EXPONENTIAL_LOGARITHMIC,
                description="The natural logarithm function, inverse of the exponential function with base e.",
                domain="(0, ∞)",
                range="All real numbers",
                properties=["Continuous", "Strictly increasing"],
                related_functions=["Exponential Function"]
            ),
            MathematicalFunction(
                name="Common Logarithm",
                symbol="f(x) = log₁₀(x)",
                category=MathematicalFunctionCategory.EXPONENTIAL_LOGARITHMIC,
                description="The logarithm function with base 10.",
                domain="(0, ∞)",
                range="All real numbers",
                properties=["Continuous", "Strictly increasing"],
                related_functions=["Inverse Exponential Function"]
            ),
            MathematicalFunction(
                name="Binary Logarithm",
                symbol="f(x) = log₂(x)",
                category=MathematicalFunctionCategory.EXPONENTIAL_LOGARITHMIC,
                description="The logarithm function with base 2.",
                domain="(0, ∞)",
                range="All real numbers",
                properties=["Continuous", "Strictly increasing"],
                related_functions=["Inverse Exponential Function"]
            ),
            MathematicalFunction(
                name="Inverse Exponential Function",
                symbol="f(x) = logₐ(x)",
                category=MathematicalFunctionCategory.EXPONENTIAL_LOGARITHMIC,
                description="The logarithm function, inverse of the exponential function with base a.",
                domain="(0, ∞)",
                range="All real numbers",
                properties=["Continuous", "Strictly increasing"],
                related_functions=["General Exponential Function"]
            ),
        ],
        MathematicalFunctionCategory.TRIGONOMETRIC: [
            MathematicalFunction(
                name="Sine",
                symbol="f(x) = sin(x)",
                category=MathematicalFunctionCategory.TRIGONOMETRIC,
                description="Trigonometric sine function.",
                domain="All real numbers",
                range="[-1, 1]",
                properties=["Periodic", "Odd function"],
                related_functions=["Cosine", "Tangent"]
            ),
            MathematicalFunction(
                name="Cosine",
                symbol="f(x) = cos(x)",
                category=MathematicalFunctionCategory.TRIGONOMETRIC,
                description="Trigonometric cosine function.",
                domain="All real numbers",
                range="[-1, 1]",
                properties=["Periodic", "Even function"],
                related_functions=["Sine", "Tangent"]
            ),
            MathematicalFunction(
                name="Tangent",
                symbol="f(x) = tan(x)",
                category=MathematicalFunctionCategory.TRIGONOMETRIC,
                description="Trigonometric tangent function.",
                domain="x ≠ (π/2 + kπ), k ∈ ℤ",
                range="All real numbers",
                properties=["Periodic", "Odd function"],
                related_functions=["Sine", "Cosine"]
            ),
            MathematicalFunction(
                name="Cosecant",
                symbol="f(x) = csc(x)",
                category=MathematicalFunctionCategory.TRIGONOMETRIC,
                description="Trigonometric cosecant function, inverse of sine.",
                domain="x ≠ kπ, k ∈ ℤ",
                range="(-∞, -1] ∪ [1, ∞)",
                properties=["Periodic", "Odd function"],
                related_functions=["Secant", "Cotangent"]
            ),
            MathematicalFunction(
                name="Secant",
                symbol="f(x) = sec(x)",
                category=MathematicalFunctionCategory.TRIGONOMETRIC,
                description="Trigonometric secant function, inverse of cosine.",
                domain="x ≠ π/2 + kπ, k ∈ ℤ",
                range="(-∞, -1] ∪ [1, ∞)",
                properties=["Periodic", "Even function"],
                related_functions=["Cosecant", "Cotangent"]
            ),
            MathematicalFunction(
                name="Cotangent",
                symbol="f(x) = cot(x)",
                category=MathematicalFunctionCategory.TRIGONOMETRIC,
                description="Trigonometric cotangent function, inverse of tangent.",
                domain="x ≠ kπ, k ∈ ℤ",
                range="All real numbers",
                properties=["Periodic", "Even function"],
                related_functions=["Cosecant", "Secant"]
            ),
        ],
        MathematicalFunctionCategory.INVERSE_TRIGONOMETRIC: [
            MathematicalFunction(
                name="Arcsine",
                symbol="f(x) = arcsin(x)",
                category=MathematicalFunctionCategory.INVERSE_TRIGONOMETRIC,
                description="Inverse of the sine function.",
                domain="[-1, 1]",
                range="[-π/2, π/2]",
                properties=["Odd function"],
                related_functions=["Arccosine", "Arctangent"]
            ),
            MathematicalFunction(
                name="Arccosine",
                symbol="f(x) = arccos(x)",
                category=MathematicalFunctionCategory.INVERSE_TRIGONOMETRIC,
                description="Inverse of the cosine function.",
                domain="[-1, 1]",
                range="[0, π]",
                properties=["Even function"],
                related_functions=["Arcsine", "Arctangent"]
            ),
            MathematicalFunction(
                name="Arctangent",
                symbol="f(x) = arctan(x)",
                category=MathematicalFunctionCategory.INVERSE_TRIGONOMETRIC,
                description="Inverse of the tangent function.",
                domain="All real numbers",
                range="(-π/2, π/2)",
                properties=["Odd function"],
                related_functions=["Arcsine", "Arccosine"]
            ),
            MathematicalFunction(
                name="Arccosecant",
                symbol="f(x) = arccsc(x)",
                category=MathematicalFunctionCategory.INVERSE_TRIGONOMETRIC,
                description="Inverse of the cosecant function.",
                domain="x ≤ -1 or x ≥ 1",
                range="[-π/2, 0) ∪ (0, π/2]",
                properties=["Odd function"],
                related_functions=["Arcsecant", "Arccotangent"]
            ),
            MathematicalFunction(
                name="Arcsecant",
                symbol="f(x) = arcsec(x)",
                category=MathematicalFunctionCategory.INVERSE_TRIGONOMETRIC,
                description="Inverse of the secant function.",
                domain="x ≤ -1 or x ≥ 1",
                range="[0, π/2) ∪ (π/2, π]",
                properties=["Even function"],
                related_functions=["Arccosecant", "Arccotangent"]
            ),
            MathematicalFunction(
                name="Arccotangent",
                symbol="f(x) = arccot(x)",
                category=MathematicalFunctionCategory.INVERSE_TRIGONOMETRIC,
                description="Inverse of the cotangent function.",
                domain="All real numbers",
                range="(0, π)",
                properties=["Odd function"],
                related_functions=["Arccosecant", "Arcsecant"]
            ),
        ],
        MathematicalFunctionCategory.HYPERBOLIC: [
            MathematicalFunction(
                name="Hyperbolic Sine",
                symbol="f(x) = sinh(x)",
                category=MathematicalFunctionCategory.HYPERBOLIC,
                description="Hyperbolic sine function.",
                domain="All real numbers",
                range="(-∞, ∞)",
                properties=["Odd function"],
                related_functions=["Hyperbolic Cosine", "Hyperbolic Tangent"]
            ),
            MathematicalFunction(
                name="Hyperbolic Cosine",
                symbol="f(x) = cosh(x)",
                category=MathematicalFunctionCategory.HYPERBOLIC,
                description="Hyperbolic cosine function.",
                domain="All real numbers",
                range="[1, ∞)",
                properties=["Even function"],
                related_functions=["Hyperbolic Sine", "Hyperbolic Tangent"]
            ),
            MathematicalFunction(
                name="Hyperbolic Tangent",
                symbol="f(x) = tanh(x)",
                category=MathematicalFunctionCategory.HYPERBOLIC,
                description="Hyperbolic tangent function.",
                domain="All real numbers",
                range="(-1, 1)",
                properties=["Odd function"],
                related_functions=["Hyperbolic Sine", "Hyperbolic Cosine"]
            ),
            MathematicalFunction(
                name="Hyperbolic Cosecant",
                symbol="f(x) = csch(x)",
                category=MathematicalFunctionCategory.HYPERBOLIC,
                description="Hyperbolic cosecant function, inverse of hyperbolic sine.",
                domain="x ≠ 0",
                range="(-∞, -1] ∪ [1, ∞)",
                properties=["Odd function"],
                related_functions=["Hyperbolic Secant", "Hyperbolic Cotangent"]
            ),
            MathematicalFunction(
                name="Hyperbolic Secant",
                symbol="f(x) = sech(x)",
                category=MathematicalFunctionCategory.HYPERBOLIC,
                description="Hyperbolic secant function, inverse of hyperbolic cosine.",
                domain="All real numbers",
                range="(0, 1]",
                properties=["Even function"],
                related_functions=["Hyperbolic Cosecant", "Hyperbolic Cotangent"]
            ),
            MathematicalFunction(
                name="Hyperbolic Cotangent",
                symbol="f(x) = coth(x)",
                category=MathematicalFunctionCategory.HYPERBOLIC,
                description="Hyperbolic cotangent function, inverse of hyperbolic tangent.",
                domain="x ≠ 0",
                range="(-∞, -1) ∪ (1, ∞)",
                properties=["Odd function"],
                related_functions=["Hyperbolic Cosecant", "Hyperbolic Secant"]
            ),
        ],
        MathematicalFunctionCategory.INVERSE_HYPERBOLIC: [
            MathematicalFunction(
                name="Area Hyperbolic Sine",
                symbol="f(x) = arsinh(x)",
                category=MathematicalFunctionCategory.INVERSE_HYPERBOLIC,
                description="Inverse of the hyperbolic sine function.",
                domain="All real numbers",
                range="All real numbers",
                properties=["Odd function"],
                related_functions=["Area Hyperbolic Cosine", "Area Hyperbolic Tangent"]
            ),
            MathematicalFunction(
                name="Area Hyperbolic Cosine",
                symbol="f(x) = arcosh(x)",
                category=MathematicalFunctionCategory.INVERSE_HYPERBOLIC,
                description="Inverse of the hyperbolic cosine function.",
                domain="[1, ∞)",
                range="[0, ∞)",
                properties=["Even function"],
                related_functions=["Area Hyperbolic Sine", "Area Hyperbolic Tangent"]
            ),
            MathematicalFunction(
                name="Area Hyperbolic Tangent",
                symbol="f(x) = artanh(x)",
                category=MathematicalFunctionCategory.INVERSE_HYPERBOLIC,
                description="Inverse of the hyperbolic tangent function.",
                domain="(-1, 1)",
                range="(-∞, ∞)",
                properties=["Odd function"],
                related_functions=["Area Hyperbolic Sine", "Area Hyperbolic Cosine"]
            ),
            MathematicalFunction(
                name="Area Hyperbolic Cosecant",
                symbol="f(x) = archcsch(x)",
                category=MathematicalFunctionCategory.INVERSE_HYPERBOLIC,
                description="Inverse of the hyperbolic cosecant function.",
                domain="x ≠ 0",
                range="All real numbers except zero",
                properties=["Odd function"],
                related_functions=["Area Hyperbolic Secant", "Area Hyperbolic Cotangent"]
            ),
            MathematicalFunction(
                name="Area Hyperbolic Secant",
                symbol="f(x) = archsech(x)",
                category=MathematicalFunctionCategory.INVERSE_HYPERBOLIC,
                description="Inverse of the hyperbolic secant function.",
                domain="(0, 1]",
                range="[0, ∞)",
                properties=["Even function"],
                related_functions=["Area Hyperbolic Cosecant", "Area Hyperbolic Cotangent"]
            ),
            MathematicalFunction(
                name="Area Hyperbolic Cotangent",
                symbol="f(x) = archcoth(x)",
                category=MathematicalFunctionCategory.INVERSE_HYPERBOLIC,
                description="Inverse of the hyperbolic cotangent function.",
                domain="x ≤ -1 or x ≥ 1",
                range="All real numbers",
                properties=["Odd function"],
                related_functions=["Area Hyperbolic Cosecant", "Area Hyperbolic Secant"]
            ),
        ],
        MathematicalFunctionCategory.SPECIAL: [
            MathematicalFunction(
                name="Gamma Function",
                symbol="Γ(n)",
                category=MathematicalFunctionCategory.SPECIAL,
                description="Generalization of the factorial function to complex numbers.",
                domain="All complex numbers except non-positive integers",
                range="All complex numbers except zero",
                properties=["Analytic function", "Integral representation"],
                related_functions=["Beta Function"]
            ),
            MathematicalFunction(
                name="Beta Function",
                symbol="B(x, y)",
                category=MathematicalFunctionCategory.SPECIAL,
                description="Function defined by an integral, related to gamma functions.",
                domain="Re(x) > 0 and Re(y) > 0",
                range="(0, ∞)",
                properties=["Symmetric in x and y", "Integral representation"],
                related_functions=["Gamma Function"]
            ),
            MathematicalFunction(
                name="Bessel Functions",
                symbol="Jₙ(x), Yₙ(x)",
                category=MathematicalFunctionCategory.SPECIAL,
                description="Solutions to Bessel's differential equation, used in wave propagation and static potentials.",
                domain="All real numbers",
                range="All real numbers",
                properties=["Oscillatory", "Orthogonal"],
                related_functions=["Legendre Polynomials"]
            ),
            MathematicalFunction(
                name="Legendre Polynomials",
                symbol="Pₙ(x)",
                category=MathematicalFunctionCategory.SPECIAL,
                description="Solutions to Legendre's differential equation, used in physics and engineering.",
                domain="[-1, 1]",
                range="[-1, 1]",
                properties=["Orthogonal", "Symmetric"],
                related_functions=["Bessel Functions"]
            ),
            MathematicalFunction(
                name="Error Function",
                symbol="erf(x)",
                category=MathematicalFunctionCategory.SPECIAL,
                description="Function used in probability, statistics, and partial differential equations.",
                domain="All real numbers",
                range="(-1, 1)",
                properties=["Odd function", "Integral representation"],
                related_functions=["Gamma Function"]
            ),
            MathematicalFunction(
                name="Exponential Integral",
                symbol="Ei(x)",
                category=MathematicalFunctionCategory.SPECIAL,
                description="Function representing the integral of (e^t)/t from -∞ to x.",
                domain="x > 0",
                range="All real numbers",
                properties=["Non-elementary function", "Integral representation"],
                related_functions=["Gamma Function"]
            ),
        ],
        MathematicalFunctionCategory.ABSOLUTE_PIECEWISE: [
            MathematicalFunction(
                name="Absolute Value Function",
                symbol="f(x) = |x|",
                category=MathematicalFunctionCategory.ABSOLUTE_PIECEWISE,
                description="Function that returns the non-negative value of x.",
                domain="All real numbers",
                range="[0, ∞)",
                properties=["Even function", "Continuous"],
                related_functions=["Heaviside Step Function"]
            ),
            MathematicalFunction(
                name="Heaviside Step Function",
                symbol="H(x)",
                category=MathematicalFunctionCategory.ABSOLUTE_PIECEWISE,
                description="Function that is zero for negative arguments and one for positive arguments.",
                domain="All real numbers",
                range="{0, 1}",
                properties=["Discontinuous at x=0", "Used in signal processing"],
                related_functions=["Sign Function"]
            ),
            MathematicalFunction(
                name="Sign Function",
                symbol="sgn(x)",
                category=MathematicalFunctionCategory.ABSOLUTE_PIECEWISE,
                description="Function that extracts the sign of a real number.",
                domain="All real numbers except x=0",
                range="{-1, 1}",
                properties=["Odd function", "Discontinuous at x=0"],
                related_functions=["Heaviside Step Function"]
            ),
        ],
        MathematicalFunctionCategory.MISCELLANEOUS: [
            MathematicalFunction(
                name="Floor Function",
                symbol="⌊x⌋",
                category=MathematicalFunctionCategory.MISCELLANEOUS,
                description="Greatest integer less than or equal to x.",
                domain="All real numbers",
                range="All integers",
                properties=["Non-continuous", "Integer output"],
                related_functions=["Ceiling Function"]
            ),
            MathematicalFunction(
                name="Ceiling Function",
                symbol="⌈x⌉",
                category=MathematicalFunctionCategory.MISCELLANEOUS,
                description="Smallest integer greater than or equal to x.",
                domain="All real numbers",
                range="All integers",
                properties=["Non-continuous", "Integer output"],
                related_functions=["Floor Function"]
            ),
            MathematicalFunction(
                name="Round Function",
                symbol="round(x)",
                category=MathematicalFunctionCategory.MISCELLANEOUS,
                description="Rounds x to the nearest integer.",
                domain="All real numbers",
                range="All integers",
                properties=["Depends on rounding rule", "Continuous"],
                related_functions=["Floor Function", "Ceiling Function"]
            ),
            MathematicalFunction(
                name="Min Function",
                symbol="min(a, b)",
                category=MathematicalFunctionCategory.MISCELLANEOUS,
                description="Returns the smaller of a and b.",
                domain="a, b ∈ ℝ",
                range="min(a, b)",
                properties=["Associative", "Commutative"],
                related_functions=["Max Function"]
            ),
            MathematicalFunction(
                name="Max Function",
                symbol="max(a, b)",
                category=MathematicalFunctionCategory.MISCELLANEOUS,
                description="Returns the larger of a and b.",
                domain="a, b ∈ ℝ",
                range="max(a, b)",
                properties=["Associative", "Commutative"],
                related_functions=["Min Function"]
            ),
        ],
    }

    @staticmethod
    def get_functions_by_category(category: MathematicalFunctionCategory) -> List[MathematicalFunction]:
        """
        Retrieve all mathematical functions under a certain category.
        """
        return MathematicalFunctions.functions.get(category, [])

    @staticmethod
    def to_json() -> str:
        """
        Convert the entire 'functions' dictionary to a JSON string.
        """
        serializable_dict = {}
        for category, funcs in MathematicalFunctions.functions.items():
            serializable_dict[category.name] = [asdict(func) for func in funcs]
        return json.dumps(serializable_dict, indent=4, cls=UniversalJSONEncoder)

    @staticmethod
    def from_json(json_str: str) -> 'MathematicalFunctions':
        """
        Populate the class's 'functions' dictionary from a JSON string.
        """
        data = json.loads(json_str, object_hook=universal_json_decoder)
        for category_name, funcs in data.items():
            try:
                category_enum = MathematicalFunctionCategory[category_name]
                MathematicalFunctions.functions[category_enum] = [
                    MathematicalFunction.from_dict(func) for func in funcs
                ]
            except KeyError:
                logging.warning(f"Unknown category '{category_name}' found in JSON and will be skipped.")
        return MathematicalFunctions

    @staticmethod
    def add_function(func: MathematicalFunction) -> None:
        """
        Add a new function to the relevant category if it does not already exist.
        """
        category = func.category
        if category not in MathematicalFunctions.functions:
            MathematicalFunctions.functions[category] = []
        if any(existing_func.name == func.name for existing_func in MathematicalFunctions.functions[category]):
            logging.error(f"Function '{func.name}' already exists in category '{category.name}'.")
            raise ValueError(f"Function '{func.name}' already exists in category '{category.name}'.")
        MathematicalFunctions.functions[category].append(func)
        logging.info(f"Added function '{func.name}' successfully.")

    @staticmethod
    def remove_function(category: MathematicalFunctionCategory, function_name: str) -> None:
        """
        Remove a function from a specific category by name.
        """
        if category not in MathematicalFunctions.functions:
            logging.error(f"Category '{category.name}' does not exist.")
            raise KeyError(f"Category '{category.name}' does not exist.")

        funcs = MathematicalFunctions.functions[category]
        for func in funcs:
            if func.name == function_name:
                funcs.remove(func)
                logging.info(f"Removed function '{function_name}' from category '{category.name}' successfully.")
                return
        logging.error(f"Function '{function_name}' not found in category '{category.name}'.")
        raise KeyError(f"Function '{function_name}' not found in category '{category.name}'.")

    @staticmethod
    def modify_function(category: MathematicalFunctionCategory, function_name: str, **kwargs) -> None:
        """
        Modify an existing mathematical function's attributes.
        """
        if category not in MathematicalFunctions.functions:
            logging.error(f"Category '{category.name}' does not exist.")
            raise KeyError(f"Category '{category.name}' does not exist.")

        for idx, func in enumerate(MathematicalFunctions.functions[category]):
            if func.name == function_name:
                updated_func = MathematicalFunction(
                    name=func.name,
                    symbol=kwargs.get('symbol', func.symbol),
                    category=func.category,
                    description=kwargs.get('description', func.description),
                    domain=kwargs.get('domain', func.domain),
                    range=kwargs.get('range', func.range),
                    properties=kwargs.get('properties', func.properties),
                    related_functions=kwargs.get('related_functions', func.related_functions)
                )
                MathematicalFunctions.functions[category][idx] = updated_func
                logging.info(f"Modified function '{function_name}' in category '{category.name}' successfully.")
                return

        logging.error(f"Function '{function_name}' not found in category '{category.name}'.")
        raise KeyError(f"Function '{function_name}' not found in category '{category.name}'.")

    @staticmethod
    def list_all_functions() -> List[MathematicalFunction]:
        """
        List all available mathematical functions across all categories.
        """
        all_funcs = []
        for funcs in MathematicalFunctions.functions.values():
            all_funcs.extend(funcs)
        return all_funcs

    @staticmethod
    def search_functions(keyword: str) -> List[MathematicalFunction]:
        """
        Search for functions by keyword in either the name or description fields.
        """
        keyword_lower = keyword.lower()
        return [
            func for funcs in MathematicalFunctions.functions.values() for func in funcs
            if keyword_lower in func.name.lower() or keyword_lower in func.description.lower()
        ]


# =============================================================================
# 5. Irreducible Logic Symbols/Structures
# =============================================================================

class LogicOperator(Enum):
    """
    Enum representing irreducible logic symbols/structures (non-composite).
    """

    # Basic Logical Operators
    AND = auto()
    OR = auto()
    NOT = auto()
    XOR = auto()

    # Derived Logical Operators
    NAND = auto()
    NOR = auto()
    XNOR = auto()

    # Logical Quantifiers
    UNIVERSAL_QUANTIFIER = auto()
    EXISTENTIAL_QUANTIFIER = auto()

    # Implication and Equivalence
    IMPLICATION = auto()
    BICONDITIONAL = auto()

    # Parentheses and Grouping Symbols
    LEFT_PARENTHESIS = auto()
    RIGHT_PARENTHESIS = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()

    # Additional Symbols
    TAUTOLOGY = auto()
    CONTRADICTION = auto()
    MATERIAL_NONIMPLICATION = auto()

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class LogicSymbol:
    """
    Dataclass representing a single logic symbol/operator, with optional precedence and associativity.
    """
    operator: LogicOperator
    symbol: str
    description: str
    precedence: Optional[int] = None
    associativity: Optional[str] = None  # 'Left', 'Right', or 'None'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'LogicSymbol':
        """
        Create a LogicSymbol instance from a dictionary.
        """
        operator = LogicOperator[data['operator']]
        return LogicSymbol(
            operator=operator,
            symbol=data['symbol'],
            description=data.get('description', ""),
            precedence=data.get('precedence'),
            associativity=data.get('associativity')
        )


class IrreducibleLogicSymbols:
    """
    Class encapsulating a list of all irreducible logic symbols in the system.
    """
    symbols: List[LogicSymbol] = [
        LogicSymbol(operator=LogicOperator.AND, symbol="∧", description="Logical AND operation.", precedence=3, associativity="Left"),
        LogicSymbol(operator=LogicOperator.OR, symbol="∨", description="Logical OR operation.", precedence=2, associativity="Left"),
        LogicSymbol(operator=LogicOperator.NOT, symbol="¬", description="Logical NOT operation.", precedence=4, associativity="Right"),
        LogicSymbol(operator=LogicOperator.XOR, symbol="⊕", description="Logical exclusive OR operation.", precedence=2, associativity="Left"),
        LogicSymbol(operator=LogicOperator.NAND, symbol="↑", description="Logical NAND operation.", precedence=3, associativity="Left"),
        LogicSymbol(operator=LogicOperator.NOR, symbol="↓", description="Logical NOR operation.", precedence=2, associativity="Left"),
        LogicSymbol(operator=LogicOperator.XNOR, symbol="≡", description="Logical XNOR operation.", precedence=2, associativity="Left"),
        LogicSymbol(operator=LogicOperator.UNIVERSAL_QUANTIFIER, symbol="∀", description="Universal quantifier (for all).", precedence=1, associativity="Right"),
        LogicSymbol(operator=LogicOperator.EXISTENTIAL_QUANTIFIER, symbol="∃", description="Existential quantifier (there exists).", precedence=1, associativity="Right"),
        LogicSymbol(operator=LogicOperator.IMPLICATION, symbol="→", description="Logical implication (if...then...).", precedence=1, associativity="Right"),
        LogicSymbol(operator=LogicOperator.BICONDITIONAL, symbol="↔", description="Logical biconditional (iff).", precedence=1, associativity="Left"),
        LogicSymbol(operator=LogicOperator.LEFT_PARENTHESIS, symbol="(", description="Left parenthesis for grouping.", precedence=5, associativity="None"),
        LogicSymbol(operator=LogicOperator.RIGHT_PARENTHESIS, symbol=")", description="Right parenthesis for grouping.", precedence=5, associativity="None"),
        LogicSymbol(operator=LogicOperator.LEFT_BRACKET, symbol="[", description="Left bracket for grouping.", precedence=5, associativity="None"),
        LogicSymbol(operator=LogicOperator.RIGHT_BRACKET, symbol="]", description="Right bracket for grouping.", precedence=5, associativity="None"),
        LogicSymbol(operator=LogicOperator.TAUTOLOGY, symbol="⊤", description="A statement that is always true.", precedence=1, associativity="None"),
        LogicSymbol(operator=LogicOperator.CONTRADICTION, symbol="⊥", description="A statement that is always false.", precedence=1, associativity="None"),
        LogicSymbol(operator=LogicOperator.MATERIAL_NONIMPLICATION, symbol="↛", description="Material Nonimplication (not implies).", precedence=1, associativity="Right"),
    ]

    @staticmethod
    def get_basic_logical_operators() -> List[LogicSymbol]:
        """
        Retrieve only the basic logical operators (AND, OR, NOT, XOR).
        """
        basic_ops = {LogicOperator.AND, LogicOperator.OR, LogicOperator.NOT, LogicOperator.XOR}
        return [s for s in IrreducibleLogicSymbols.symbols if s.operator in basic_ops]

    @staticmethod
    def get_logic_symbol_by_operator(operator: LogicOperator) -> Optional[LogicSymbol]:
        """
        Retrieve a specific LogicSymbol by its operator enum.

        Returns:
            LogicSymbol or None
        """
        for symbol in IrreducibleLogicSymbols.symbols:
            if symbol.operator == operator:
                return symbol
        return None

    @staticmethod
    def to_json() -> str:
        """
        Serialize the entire symbol list to JSON.
        """
        serializable_list = [asdict(symbol) for symbol in IrreducibleLogicSymbols.symbols]
        return json.dumps(serializable_list, indent=4, cls=UniversalJSONEncoder)

    @staticmethod
    def from_json(json_str: str) -> 'IrreducibleLogicSymbols':
        """
        Convert a JSON string of logic symbols into the existing class list, skipping duplicates.
        """
        data = json.loads(json_str, object_hook=universal_json_decoder)
        for item in data:
            try:
                symbol = LogicSymbol.from_dict(item)
                if any(existing.symbol == symbol.symbol for existing in IrreducibleLogicSymbols.symbols):
                    logging.warning(f"Duplicate logic symbol '{symbol.symbol}' found in JSON and will be skipped.")
                    continue
                IrreducibleLogicSymbols.symbols.append(symbol)
                logging.info(f"Added logic symbol '{symbol.operator}' successfully.")
            except KeyError as e:
                logging.warning(f"Invalid data for LogicSymbol: {e}. Entry will be skipped.")
        return IrreducibleLogicSymbols

    @staticmethod
    def add_logic_symbol(symbol: LogicSymbol) -> None:
        """
        Add a new logic symbol if it does not already exist.
        """
        if any(existing.symbol == symbol.symbol for existing in IrreducibleLogicSymbols.symbols):
            logging.error(f"Logic symbol '{symbol.symbol}' already exists.")
            raise ValueError(f"Logic symbol '{symbol.symbol}' already exists.")
        IrreducibleLogicSymbols.symbols.append(symbol)
        logging.info(f"Added logic symbol '{symbol.operator.name}' successfully.")

    @staticmethod
    def remove_logic_symbol(operator: LogicOperator) -> None:
        """
        Remove an existing logic symbol by operator.
        """
        for symbol in IrreducibleLogicSymbols.symbols:
            if symbol.operator == operator:
                IrreducibleLogicSymbols.symbols.remove(symbol)
                logging.info(f"Removed logic symbol '{operator.name}' successfully.")
                return
        logging.error(f"Logic symbol '{operator.name}' does not exist.")
        raise KeyError(f"Logic symbol '{operator.name}' does not exist.")

    @staticmethod
    def modify_logic_symbol(operator: LogicOperator, **kwargs) -> None:
        """
        Modify an existing logic symbol's fields.
        """
        for idx, sym in enumerate(IrreducibleLogicSymbols.symbols):
            if sym.operator == operator:
                updated_symbol = LogicSymbol(
                    operator=sym.operator,
                    symbol=kwargs.get('symbol', sym.symbol),
                    description=kwargs.get('description', sym.description),
                    precedence=kwargs.get('precedence', sym.precedence),
                    associativity=kwargs.get('associativity', sym.associativity)
                )
                # Prevent symbol duplication if 'symbol' was changed
                if updated_symbol.symbol != sym.symbol and any(existing.symbol == updated_symbol.symbol for existing in IrreducibleLogicSymbols.symbols):
                    logging.error(f"Logic symbol '{updated_symbol.symbol}' already exists.")
                    raise ValueError(f"Logic symbol '{updated_symbol.symbol}' already exists.")
                IrreducibleLogicSymbols.symbols[idx] = updated_symbol
                logging.info(f"Modified logic symbol '{operator.name}' successfully.")
                return
        logging.error(f"Logic symbol '{operator.name}' does not exist.")
        raise KeyError(f"Logic symbol '{operator.name}' does not exist.")

    @staticmethod
    def list_all_logic_symbols() -> List[LogicSymbol]:
        """
        Return a copy of the entire logic symbol list.
        """
        return IrreducibleLogicSymbols.symbols.copy()

    @staticmethod
    def search_logic_symbols(keyword: str) -> List[LogicSymbol]:
        """
        Search for logic symbols by keyword match in operator name or description.
        """
        keyword_lower = keyword.lower()
        return [
            s for s in IrreducibleLogicSymbols.symbols
            if keyword_lower in s.operator.name.lower() or keyword_lower in s.description.lower()
        ]


# =============================================================================
# 6. Irreducible English Characters, Phonics, and Phonemes
# =============================================================================

class EnglishCharacter(Enum):
    """
    Enum representing irreducible English characters,
    including uppercase and lowercase letters.
    """
    # Uppercase
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    J = auto()
    K = auto()
    L = auto()
    M = auto()
    N = auto()
    O = auto()
    P = auto()
    Q = auto()
    R = auto()
    S = auto()
    T = auto()
    U = auto()
    V = auto()
    W = auto()
    X = auto()
    Y = auto()
    Z = auto()

    # Lowercase
    a = auto()
    b = auto()
    c = auto()
    d = auto()
    e = auto()
    f = auto()
    g = auto()
    h = auto()
    i = auto()
    j = auto()
    k = auto()
    l = auto()
    m = auto()
    n = auto()
    o = auto()
    p = auto()
    q = auto()
    r = auto()
    s = auto()
    t = auto()
    u = auto()
    v = auto()
    w = auto()
    x = auto()
    y = auto()
    z = auto()

    def __str__(self):
        return self.name


class PhonemeCategory(Enum):
    """
    Enum representing categories of phonemes in English (Vowel, Consonant, Diphthong, Triphthong).
    """
    VOWEL = 'Vowel'
    DIPHTHONG = 'Diphthong'
    TRIPHTHONG = 'Triphthong'
    CONSONANT = 'Consonant'

    @staticmethod
    def from_str(label: str) -> 'PhonemeCategory':
        """
        Convert a string to the corresponding PhonemeCategory enum, with flexible matching.
        """
        if not isinstance(label, str):
            raise TypeError(f"Label must be a string, got {type(label).__name__} instead.")

        normalized_label = label.strip().lower()
        mapping = {
            'vowel': PhonemeCategory.VOWEL,
            'consonant': PhonemeCategory.CONSONANT,
            'diphthong': PhonemeCategory.DIPHTHONG,
            'triphthong': PhonemeCategory.TRIPHTHONG,
            'triphthongs': PhonemeCategory.TRIPHTHONG  # Plural for convenience
        }
        for key in mapping:
            if re.fullmatch(key, normalized_label):
                return mapping[key]

        valid_categories = ', '.join(f"'{cat.value}'" for cat in PhonemeCategory)
        raise ValueError(
            f"Unknown phoneme category: '{label}'. Valid categories: {valid_categories}."
        )


@dataclass(frozen=True)
class Phoneme:
    """
    Dataclass representing a single English phoneme with enforced validation for the symbol field.
    """
    symbol: str
    example: str
    category: PhonemeCategory
    description: str = ""
    ipa: Optional[str] = None
    example_sentence: Optional[str] = None

    def __post_init__(self):
        if not self.symbol.startswith("/") or not self.symbol.endswith("/"):
            raise ValueError(f"Symbol '{self.symbol}' must be enclosed in slashes, e.g., '/iː/'.")
        if not self.ipa:
            object.__setattr__(self, 'ipa', self.symbol.strip("/"))
        if self.category not in PhonemeCategory:
            raise ValueError(f"Invalid category '{self.category}' for Phoneme.")

    def __str__(self):
        return f"{self.symbol} ({self.ipa}) - {self.category.value}: {self.example}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Phoneme':
        """
        Rebuild a Phoneme instance from a dictionary, converting 'category' if it is a string.
        """
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = PhonemeCategory.from_str(data['category'])
        return Phoneme(
            symbol=data['symbol'],
            example=data['example'],
            category=data['category'],
            description=data.get('description', ""),
            ipa=data.get('ipa'),
            example_sentence=data.get('example_sentence')
        )


class IrreducibleEnglishPhonemes:
    """
    Class encapsulating concurrency-safe operations on a list of English phonemes.
    """
    _phonemes: List[Phoneme] = [
        # A. Vowel Phonemes
        Phoneme(symbol="/iː/", example='"see"', category=PhonemeCategory.VOWEL,
                description="Long 'ee' sound as in 'see'.",
                ipa="iː",
                example_sentence='"She can see the stars."'),
        Phoneme(symbol="/ɪ/", example='"sit"', category=PhonemeCategory.VOWEL,
                description="Short 'i' sound as in 'sit'.",
                ipa="ɪ",
                example_sentence='"He likes to sit quietly."'),
        Phoneme(symbol="/e/", example='"bed"', category=PhonemeCategory.VOWEL,
                description="Short 'e' sound as in 'bed'.",
                ipa="e",
                example_sentence='"She lies in bed all day."'),
        Phoneme(symbol="/æ/", example='"cat"', category=PhonemeCategory.VOWEL,
                description="Short 'a' sound as in 'cat'.",
                ipa="æ",
                example_sentence='"The cat sat on the mat."'),
        Phoneme(symbol="/ɑː/", example='"father"', category=PhonemeCategory.VOWEL,
                description="Long 'a' sound as in 'father'.",
                ipa="ɑː",
                example_sentence='"He is a loving father."'),
        Phoneme(symbol="/ɒ/", example='"hot" (British English)', category=PhonemeCategory.VOWEL,
                description="Short 'o' sound as in 'hot' (British English).",
                ipa="ɒ",
                example_sentence='"It is hot outside."'),
        Phoneme(symbol="/ɔː/", example='"law"', category=PhonemeCategory.VOWEL,
                description="Long 'aw' sound as in 'law'.",
                ipa="ɔː",
                example_sentence='"The law is clear."'),
        Phoneme(symbol="/ʊ/", example='"put"', category=PhonemeCategory.VOWEL,
                description="Short 'u' sound as in 'put'.",
                ipa="ʊ",
                example_sentence='"Please put the book on the table."'),
        Phoneme(symbol="/uː/", example='"food"', category=PhonemeCategory.VOWEL,
                description="Long 'oo' sound as in 'food'.",
                ipa="uː",
                example_sentence='"They enjoyed the delicious food."'),
        Phoneme(symbol="/ʌ/", example='"cup"', category=PhonemeCategory.VOWEL,
                description="Short 'u' sound as in 'cup'.",
                ipa="ʌ",
                example_sentence='"He drinks from a cup."'),
        Phoneme(symbol="/ɜːr/", example='"bird" (British English)', category=PhonemeCategory.VOWEL,
                description="R-colored vowel as in 'bird' (British English).",
                ipa="ɜːr",
                example_sentence='"The bird is singing."'),
        Phoneme(symbol="/ər/", example='"water" (American English)', category=PhonemeCategory.VOWEL,
                description="R-colored vowel as in 'water' (American English).",
                ipa="ər",
                example_sentence='"She poured water into the glass."'),

        # B. Diphthongs
        Phoneme(symbol="/aɪ/", example='"my"', category=PhonemeCategory.DIPHTHONG,
                description="Diphthong combining 'a' and 'ɪ' as in 'my'.",
                ipa="aɪ",
                example_sentence='"This is my book."'),
        Phoneme(symbol="/aʊ/", example='"now"', category=PhonemeCategory.DIPHTHONG,
                description="Diphthong combining 'a' and 'ʊ' as in 'now'.",
                ipa="aʊ",
                example_sentence='"I need to leave now."'),
        Phoneme(symbol="/ɔɪ/", example='"boy"', category=PhonemeCategory.DIPHTHONG,
                description="Diphthong combining 'ɔ' and 'ɪ' as in 'boy'.",
                ipa="ɔɪ",
                example_sentence='"The boy is playing."'),
        Phoneme(symbol="/eɪ/", example='"day"', category=PhonemeCategory.DIPHTHONG,
                description="Diphthong combining 'e' and 'ɪ' as in 'day'.",
                ipa="eɪ",
                example_sentence='"It is a sunny day."'),
        Phoneme(symbol="/oʊ/", example='"go"', category=PhonemeCategory.DIPHTHONG,
                description="Diphthong combining 'o' and 'ʊ' as in 'go'.",
                ipa="oʊ",
                example_sentence='"They decided to go home."'),

        # C. Triphthongs
        Phoneme(symbol="/aɪə/", example='"fire"', category=PhonemeCategory.TRIPHTHONG,
                description="Triphthong combining 'aɪ' and 'ə' as in 'fire'.",
                ipa="aɪə",
                example_sentence='"The fire is burning brightly."'),
        Phoneme(symbol="/aʊə/", example='"flower"', category=PhonemeCategory.TRIPHTHONG,
                description="Triphthong combining 'aʊ' and 'ə' as in 'flower'.",
                ipa="aʊə",
                example_sentence='"She picked a beautiful flower."'),
        Phoneme(symbol="/eɪə/", example='"player"', category=PhonemeCategory.TRIPHTHONG,
                description="Triphthong combining 'eɪ' and 'ə' as in 'player'.",
                ipa="eɪə",
                example_sentence='"The player scored a goal."'),

        # D. Consonant Phonemes
        Phoneme(symbol="/p/", example='"pat"', category=PhonemeCategory.CONSONANT,
                description="Voiceless bilabial plosive as in 'pat'.",
                ipa="p",
                example_sentence='"Pat is reading a book."'),
        Phoneme(symbol="/b/", example='"bat"', category=PhonemeCategory.CONSONANT,
                description="Voiced bilabial plosive as in 'bat'.",
                ipa="b",
                example_sentence='"The bat flew away."'),
        Phoneme(symbol="/t/", example='"top"', category=PhonemeCategory.CONSONANT,
                description="Voiceless alveolar plosive as in 'top'.",
                ipa="t",
                example_sentence='"She reached the top of the hill."'),
        Phoneme(symbol="/d/", example='"dog"', category=PhonemeCategory.CONSONANT,
                description="Voiced alveolar plosive as in 'dog'.",
                ipa="d",
                example_sentence='"The dog is barking."'),
        Phoneme(symbol="/k/", example='"cat"', category=PhonemeCategory.CONSONANT,
                description="Voiceless velar plosive as in 'cat'.",
                ipa="k",
                example_sentence='"The cat is sleeping."'),
        Phoneme(symbol="/g/", example='"go"', category=PhonemeCategory.CONSONANT,
                description="Voiced velar plosive as in 'go'.",
                ipa="g",
                example_sentence='"They decided to go home."'),
        Phoneme(symbol="/f/", example='"fish"', category=PhonemeCategory.CONSONANT,
                description="Voiceless labiodental fricative as in 'fish'.",
                ipa="f",
                example_sentence='"The fish swims in the pond."'),
        Phoneme(symbol="/v/", example='"voice"', category=PhonemeCategory.CONSONANT,
                description="Voiced labiodental fricative as in 'voice'.",
                ipa="v",
                example_sentence='"She has a beautiful voice."'),
        Phoneme(symbol="/θ/", example='"think"', category=PhonemeCategory.CONSONANT,
                description="Voiceless dental fricative as in 'think'.",
                ipa="θ",
                example_sentence='"I think it is a good idea."'),
        Phoneme(symbol="/ð/", example='"this"', category=PhonemeCategory.CONSONANT,
                description="Voiced dental fricative as in 'this'.",
                ipa="ð",
                example_sentence='"This is my house."'),
        Phoneme(symbol="/s/", example='"sun"', category=PhonemeCategory.CONSONANT,
                description="Voiceless alveolar fricative as in 'sun'.",
                ipa="s",
                example_sentence='"The sun is shining."'),
        Phoneme(symbol="/z/", example='"zoo"', category=PhonemeCategory.CONSONANT,
                description="Voiced alveolar fricative as in 'zoo'.",
                ipa="z",
                example_sentence='"We are going to the zoo."'),
        Phoneme(symbol="/ʃ/", example='"she"', category=PhonemeCategory.CONSONANT,
                description="Voiceless postalveolar fricative as in 'she'.",
                ipa="ʃ",
                example_sentence='"She is reading a book."'),
        Phoneme(symbol="/ʒ/", example='"vision"', category=PhonemeCategory.CONSONANT,
                description="Voiced postalveolar fricative as in 'vision'.",
                ipa="ʒ",
                example_sentence='"The vision is clear."'),
        Phoneme(symbol="/tʃ/", example='"chair"', category=PhonemeCategory.CONSONANT,
                description="Voiceless postalveolar affricate as in 'chair'.",
                ipa="tʃ",
                example_sentence='"She sat on the chair."'),
        Phoneme(symbol="/dʒ/", example='"jump"', category=PhonemeCategory.CONSONANT,
                description="Voiced postalveolar affricate as in 'jump'.",
                ipa="dʒ",
                example_sentence='"He decided to jump over the fence."'),
        Phoneme(symbol="/m/", example='"man"', category=PhonemeCategory.CONSONANT,
                description="Bilabial nasal as in 'man'.",
                ipa="m",
                example_sentence='"The man is walking."'),
        Phoneme(symbol="/n/", example='"no"', category=PhonemeCategory.CONSONANT,
                description="Alveolar nasal as in 'no'.",
                ipa="n",
                example_sentence='"No, I do not agree."'),
        Phoneme(symbol="/ŋ/", example='"sing"', category=PhonemeCategory.CONSONANT,
                description="Velar nasal as in 'sing'.",
                ipa="ŋ",
                example_sentence='"She is singing a song."'),
        Phoneme(symbol="/h/", example='"hat"', category=PhonemeCategory.CONSONANT,
                description="Voiceless glottal fricative as in 'hat'.",
                ipa="h",
                example_sentence='"He wears a hat."'),
        Phoneme(symbol="/l/", example='"let"', category=PhonemeCategory.CONSONANT,
                description="Alveolar lateral approximant as in 'let'.",
                ipa="l",
                example_sentence='"Let us begin the meeting."'),
        Phoneme(symbol="/r/", example='"red"', category=PhonemeCategory.CONSONANT,
                description="Alveolar approximant as in 'red'.",
                ipa="r",
                example_sentence='"The red apple is delicious."'),
        Phoneme(symbol="/j/", example='"yes"', category=PhonemeCategory.CONSONANT,
                description="Palatal approximant as in 'yes'.",
                ipa="j",
                example_sentence='"Yes, I would like some coffee."'),
        Phoneme(symbol="/w/", example='"we"', category=PhonemeCategory.CONSONANT,
                description="Labio-velar approximant as in 'we'.",
                ipa="w",
                example_sentence='"We are going to the park."'),
    ]

    _lock = threading.Lock()

    @classmethod
    def get_phonemes_by_category(cls, category: PhonemeCategory) -> List[Phoneme]:
        """
        Retrieve phonemes by their category.
        """
        with cls._lock:
            return [
                phoneme for phoneme in cls._phonemes
                if phoneme.category == category
            ]

    @classmethod
    def to_json(cls) -> str:
        """
        Serialize all phonemes to JSON with concurrency safety.
        """
        with cls._lock:
            serializable_list = [phoneme.to_dict() for phoneme in cls._phonemes]
        return json.dumps(serializable_list, indent=4, cls=UniversalJSONEncoder)

    @classmethod
    def from_json(cls, json_str: str) -> 'IrreducibleEnglishPhonemes':
        """
        Deserialize JSON into the internal list of phonemes, skipping duplicates.
        """
        data = json.loads(json_str, object_hook=universal_json_decoder)
        if not isinstance(data, list):
            logging.error("JSON data must be a list of phoneme dictionaries.")
            raise ValueError("JSON data must be a list of phoneme dictionaries.")

        with cls._lock:
            for phoneme_data in data:
                try:
                    phoneme = Phoneme.from_dict(phoneme_data)
                    if any(existing.symbol == phoneme.symbol for existing in cls._phonemes):
                        logging.warning(f"Duplicate phoneme symbol '{phoneme.symbol}' found; skipping.")
                        continue
                    cls._phonemes.append(phoneme)
                    logging.info(f"Added phoneme '{phoneme.symbol}' successfully.")
                except (KeyError, ValueError) as e:
                    logging.warning(f"Invalid data for Phoneme: {e}. Entry will be skipped.")
        return cls

    @classmethod
    def add_phoneme(cls, phoneme: Phoneme) -> None:
        """
        Add a new phoneme if its symbol does not exist.
        """
        with cls._lock:
            if any(existing.symbol == phoneme.symbol for existing in cls._phonemes):
                logging.error(f"Phoneme '{phoneme.symbol}' already exists.")
                raise ValueError(f"Phoneme '{phoneme.symbol}' already exists.")
            cls._phonemes.append(phoneme)
            logging.info(f"Added phoneme '{phoneme.symbol}' successfully.")

    @classmethod
    def remove_phoneme(cls, symbol: str) -> None:
        """
        Remove a phoneme by its symbol.
        """
        with cls._lock:
            for phoneme in cls._phonemes:
                if phoneme.symbol == symbol:
                    cls._phonemes.remove(phoneme)
                    logging.info(f"Removed phoneme '{symbol}' successfully.")
                    return
            logging.error(f"Phoneme '{symbol}' does not exist.")
            raise KeyError(f"Phoneme '{symbol}' does not exist.")

    @classmethod
    def modify_phoneme(cls, symbol: str, **kwargs) -> None:
        """
        Modify specific fields of an existing phoneme, ensuring concurrency safety.
        """
        with cls._lock:
            for idx, phoneme in enumerate(cls._phonemes):
                if phoneme.symbol == symbol:
                    phoneme_dict = phoneme.to_dict()
                    for key, value in kwargs.items():
                        if key in phoneme_dict and value is not None:
                            phoneme_dict[key] = value
                    updated_phoneme = Phoneme.from_dict(phoneme_dict)
                    cls._phonemes[idx] = updated_phoneme
                    logging.info(f"Modified phoneme '{symbol}' successfully.")
                    return
            logging.error(f"Phoneme '{symbol}' does not exist.")
            raise KeyError(f"Phoneme '{symbol}' does not exist.")

    @classmethod
    def list_all_phonemes(cls) -> List[Phoneme]:
        """
        Provide a thread-safe copy of the phoneme list.
        """
        with cls._lock:
            return cls._phonemes.copy()

    @classmethod
    def search_phonemes(cls, keyword: str) -> List[Phoneme]:
        """
        Search for phonemes by a keyword in symbol or description.
        """
        keyword_lower = keyword.lower()
        with cls._lock:
            return [
                p for p in cls._phonemes
                if keyword_lower in p.symbol.lower() or keyword_lower in p.description.lower()
            ]

    @classmethod
    def sort_phonemes(cls, key: str, reverse: bool = False) -> List[Phoneme]:
        """
        Sort the phonemes based on a particular attribute key.
        """
        valid_keys = {'symbol', 'example', 'category', 'ipa', 'description', 'example_sentence'}
        if key not in valid_keys:
            logging.error(f"Invalid sort key '{key}'. Valid keys are {valid_keys}.")
            raise ValueError(f"Invalid sort key '{key}'. Valid keys are {valid_keys}.")
        with cls._lock:
            sorted_list = sorted(cls._phonemes, key=lambda x: getattr(x, key), reverse=reverse)
        logging.info(f"Phonemes sorted by '{key}' in {'descending' if reverse else 'ascending'} order.")
        return sorted_list

    @classmethod
    def reset_phonemes(cls) -> None:
        """
        Reset the phoneme list to an initial default set if desired.
        Note: The user can customize or omit this behavior.
        """
        with cls._lock:
            cls._phonemes.clear()
            # Potentially re-add any "default" phonemes here if needed.
            logging.info("Phoneme list has been reset to an empty list.")


# =============================================================================
# 7. Validator Class
# =============================================================================

class Validator:
    """
    A centralized Validator that runs checks on every class/Enum in the module,
    capturing errors and warnings. Results are reported at completion.
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_enums(self):
        """
        Validate that all enumerations have unique integer values (no duplicates).
        """
        logging.info("Validating Enums...")
        self._validate_enum_unique_values(ComputationalInstruction, "ComputationalInstruction")
        self._validate_enum_unique_values(MathematicalFunctionCategory, "MathematicalFunctionCategory")
        self._validate_enum_unique_values(LogicOperator, "LogicOperator")
        self._validate_enum_unique_values(EnglishCharacter, "EnglishCharacter")
        self._validate_enum_unique_values(PhonemeCategory, "PhonemeCategory")

    def _validate_enum_unique_values(self, enum_class: Enum, enum_name: str):
        """
        Ensure each Enum's integer values are unique.
        """
        seen = set()
        duplicates = set()
        for member in enum_class:
            if member.value in seen:
                duplicates.add(member.value)
            seen.add(member.value)

        if duplicates:
            error_message = f"Duplicate values found in Enum {enum_name}: {duplicates}"
            logging.error(error_message)
            self.errors.append(error_message)
        else:
            logging.info(f"All values in Enum {enum_name} are unique.")

    def validate_computational_instructions(self):
        """
        Ensure each instruction in the enum has a corresponding InstructionDetail.
        """
        logging.info("Validating Computational Instructions...")
        instructions = ComputationalInstructions.instructions
        enum_members = set(item for item in ComputationalInstruction)

        # Missing details
        missing_details = enum_members - set(instructions.keys())
        if missing_details:
            msg = f"Missing InstructionDetail for instructions: {[instr.name for instr in missing_details]}"
            logging.error(msg)
            self.errors.append(msg)
        else:
            logging.info("All ComputationalInstruction members have InstructionDetail entries.")

        # Extra details
        extra_details = set(instructions.keys()) - enum_members
        if extra_details:
            msg = f"Extra InstructionDetail entries not in Enum: {[instr.name for instr in extra_details]}"
            logging.warning(msg)
            self.warnings.append(msg)
        else:
            logging.info("No extra InstructionDetail entries found.")

        # Validate each detail
        instructions_no_operands = {
            ComputationalInstruction.NOP,
            ComputationalInstruction.RET,
            ComputationalInstruction.RETURN_FUNCTION,
            ComputationalInstruction.TRY,
            ComputationalInstruction.CATCH,
            ComputationalInstruction.THROW,
            ComputationalInstruction.HALT
        }
        for instruction, detail in instructions.items():
            logging.debug(f"Validating detail for {instruction.name}...")
            if not detail.description:
                err = f"Instruction '{instruction.name}' is missing a description."
                logging.error(err)
                self.errors.append(err)
            if not isinstance(detail.operands, list):
                err = f"Operands for instruction '{instruction.name}' should be a list."
                logging.error(err)
                self.errors.append(err)
            if instruction not in instructions_no_operands and not detail.operands:
                warn = f"Instruction '{instruction.name}' has no operands defined."
                logging.warning(warn)
                self.warnings.append(warn)
            if not detail.example_usage:
                warn = f"Instruction '{instruction.name}' is missing an example usage."
                logging.warning(warn)
                self.warnings.append(warn)

    def validate_mathematical_functions(self):
        """
        Ensure each category is populated and functions have valid fields.
        """
        logging.info("Validating Mathematical Functions...")
        functions = MathematicalFunctions.functions

        # Check that all categories are populated
        empty_categories = [cat.name for cat, funcs in functions.items() if not funcs]
        if empty_categories:
            msg = f"Categories with no functions: {empty_categories}"
            logging.error(msg)
            self.errors.append(msg)
        else:
            logging.info("All categories are populated with functions.")

        # Validate each category for duplicates
        for category, funcs in functions.items():
            names = set()
            symbols = set()
            dup_names = set()
            dup_symbols = set()
            for func in funcs:
                if func.name in names:
                    dup_names.add(func.name)
                else:
                    names.add(func.name)
                if func.symbol in symbols:
                    dup_symbols.add(func.symbol)
                else:
                    symbols.add(func.symbol)

                # More granular checks
                if not func.description:
                    err = f"MathematicalFunction '{func.name}' is missing a description."
                    logging.error(err)
                    self.errors.append(err)
                if not func.symbol:
                    err = f"MathematicalFunction '{func.name}' is missing a symbol."
                    logging.error(err)
                    self.errors.append(err)
                if func.category != category:
                    err = f"MathematicalFunction '{func.name}' has mismatched category {func.category} vs {category}."
                    logging.error(err)
                    self.errors.append(err)

            if dup_names:
                msg = f"Duplicate function names in {category.name}: {dup_names}"
                logging.error(msg)
                self.errors.append(msg)
            if dup_symbols:
                msg = f"Duplicate function symbols in {category.name}: {dup_symbols}"
                logging.error(msg)
                self.errors.append(msg)

    def validate_logic_symbols(self):
        """
        Validate logic symbols for duplicates and required fields.
        """
        logging.info("Validating Logic Symbols...")
        symbols = IrreducibleLogicSymbols.symbols
        symbol_strings = set()
        duplicates = set()

        for sym in symbols:
            if sym.symbol in symbol_strings:
                duplicates.add(sym.symbol)
            else:
                symbol_strings.add(sym.symbol)

            # Field checks
            if not sym.description:
                err = f"LogicSymbol '{sym.operator.name}' is missing a description."
                logging.error(err)
                self.errors.append(err)
            if not sym.symbol:
                err = f"LogicSymbol '{sym.operator.name}' is missing a symbol."
                logging.error(err)
                self.errors.append(err)
            if sym.precedence is not None and not isinstance(sym.precedence, int):
                err = f"LogicSymbol '{sym.operator.name}' has invalid precedence value."
                logging.error(err)
                self.errors.append(err)
            if sym.associativity not in {"Left", "Right", "None", None}:
                err = f"LogicSymbol '{sym.operator.name}' has invalid associativity value."
                logging.error(err)
                self.errors.append(err)

        if duplicates:
            msg = f"Duplicate logic symbols found: {duplicates}"
            logging.error(msg)
            self.errors.append(msg)
        else:
            logging.info("No duplicate logic symbols found.")

    def validate_phonemes(self):
        """
        Check phonemes for valid categories, non-empty fields, duplicates, etc.
        """
        logging.info("Validating Phonemes...")
        phonemes = IrreducibleEnglishPhonemes.list_all_phonemes()
        category_values = {cat.value.lower() for cat in PhonemeCategory}
        symbol_set = set()
        duplicates = set()

        for ph in phonemes:
            if ph.symbol in symbol_set:
                duplicates.add(ph.symbol)
            else:
                symbol_set.add(ph.symbol)

            if ph.category.value.lower() not in category_values:
                err = f"Phoneme '{ph.symbol}' has invalid category '{ph.category.value}'."
                logging.error(err)
                self.errors.append(err)
            if not ph.description:
                warn = f"Phoneme '{ph.symbol}' is missing a description."
                logging.warning(warn)
                self.warnings.append(warn)
            if not ph.example:
                warn = f"Phoneme '{ph.symbol}' is missing an example word."
                logging.warning(warn)
                self.warnings.append(warn)
            if not ph.ipa:
                warn = f"Phoneme '{ph.symbol}' is missing an IPA representation."
                logging.warning(warn)
                self.warnings.append(warn)
            if not ph.example_sentence:
                warn = f"Phoneme '{ph.symbol}' is missing an example sentence."
                logging.warning(warn)
                self.warnings.append(warn)

        if duplicates:
            msg = f"Duplicate phoneme symbols found: {duplicates}"
            logging.error(msg)
            self.errors.append(msg)
        else:
            logging.info("No duplicate phoneme symbols found.")

    def validate_serialization(self):
        """
        Validate that each major class can serialize and deserialize correctly.
        """
        logging.info("Validating Serialization for major classes...")

        # 1. ComputationalInstructions
        try:
            ci_json = ComputationalInstructions.to_json()
            json.loads(ci_json)
            logging.info("ComputationalInstructions serialization successful.")
        except Exception as e:
            msg = f"ComputationalInstructions serialization failed: {e}"
            logging.error(msg)
            self.errors.append(msg)

        # 2. MathematicalFunctions
        try:
            mf_json = MathematicalFunctions.to_json()
            json.loads(mf_json)
            logging.info("MathematicalFunctions serialization successful.")
        except Exception as e:
            msg = f"MathematicalFunctions serialization failed: {e}"
            logging.error(msg)
            self.errors.append(msg)

        # 3. IrreducibleLogicSymbols
        try:
            ls_json = IrreducibleLogicSymbols.to_json()
            json.loads(ls_json)
            logging.info("IrreducibleLogicSymbols serialization successful.")
        except Exception as e:
            msg = f"IrreducibleLogicSymbols serialization failed: {e}"
            logging.error(msg)
            self.errors.append(msg)

        # 4. IrreducibleEnglishPhonemes
        try:
            ep_json = IrreducibleEnglishPhonemes.to_json()
            json.loads(ep_json)
            logging.info("IrreducibleEnglishPhonemes serialization successful.")
        except Exception as e:
            msg = f"IrreducibleEnglishPhonemes serialization failed: {e}"
            logging.error(msg)
            self.errors.append(msg)

    def run_all_validations(self):
        """
        Run each validation method in sequence, capturing all logs, errors, and warnings.
        """
        self.validate_enums()
        self.validate_computational_instructions()
        self.validate_mathematical_functions()
        self.validate_logic_symbols()
        self.validate_phonemes()
        self.validate_serialization()

    def report(self):
        """
        Print out the final results of validation:
        - Any fatal errors are printed as ERROR logs.
        - Non-fatal issues are printed as WARNING logs.
        """
        if self.errors:
            logging.error("Validation completed with errors:")
            for error in self.errors:
                logging.error(f"- {error}")
        else:
            logging.info("No errors found. Validation passed successfully.")

        if self.warnings:
            logging.warning("Validation completed with warnings:")
            for warning in self.warnings:
                logging.warning(f"- {warning}")
        else:
            logging.info("No warnings were generated during validation.")


# =============================================================================
# 8. Main Function and Entry Point
# =============================================================================

def main():
    """
    Main function to execute all validations for the universal_components module.
    Raises SystemExit with code 1 if any errors are found, otherwise 0.
    """
    logging.info("Starting universal_components.py Validator...")
    validator = Validator()
    validator.run_all_validations()
    validator.report()
    logging.info("Validator Execution Completed.")

    if validator.errors:
        sys.exit(1)
    else:
        sys.exit(0)


# =============================================================================
# 9. Module Export
# =============================================================================

__all__ = [
    'ComputationalInstruction',
    'InstructionDetail',
    'ComputationalInstructions',
    'MathematicalFunctionCategory',
    'MathematicalFunction',
    'MathematicalFunctions',
    'LogicOperator',
    'LogicSymbol',
    'IrreducibleLogicSymbols',
    'EnglishCharacter',
    'Phoneme',
    'PhonemeCategory',
    'IrreducibleEnglishPhonemes',
    'Validator',
    'UniversalJSONEncoder',
    'JSONFormatter',
    'configure_logging'
]


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    main()
