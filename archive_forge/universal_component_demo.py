# universal_components_demo.py
"""
Universal Components Demo

A comprehensive demonstration script for the universal_components.py module.
This script programmatically exercises nearly every function, class, and interface
exposed by the universal_components program. It showcases:

1. Validation of all enumerations, classes, data, and relationships.
2. Listing, searching, adding, modifying, and removing instructions, functions,
   logic symbols, and phonemes.
3. Serialization and deserialization to/from JSON.
4. A SimpleVirtualMachine that interprets various computational instructions.
5. Logical expression evaluation with recognized logic operators.
6. Mathematical function computations using Python's math library.
7. Phoneme processing with concurrency safety, including a basic mapping from text.
8. Advanced NLP-based instruction translation (natural language → instructions).
9. PyTorch-based function approximation for sine (example of ML usage).
10. Logical sentiment analysis (combining TextBlob, NLTK, and logic).
11. Symbolic equation solving with sympy.
12. Phoneme-based text generation with an LSTM model in PyTorch.
13. Rich library usage (optional) to display the generated JSON logs in a
    neat, colorized table format.

Note: Some features require additional libraries (TextBlob, NLTK, PyTorch, Sympy, Rich).
Install as needed:
  pip install textblob nltk torch sympy rich
"""

import os
import sys
import re
import json
import logging
import operator
from typing import List, Dict, Any, Optional
import nltk
from textblob import TextBlob

# Third-party or optional imports (handled in code where relevant)
# import nltk, torch, textblob, sympy, rich etc. as needed in specific functions

# universal_components imports
from universal_components import (
    ComputationalInstruction,
    InstructionDetail,
    ComputationalInstructions,
    MathematicalFunctionCategory,
    MathematicalFunction,
    MathematicalFunctions,
    LogicOperator,
    LogicSymbol,
    IrreducibleLogicSymbols,
    EnglishCharacter,
    Phoneme,
    IrreducibleEnglishPhonemes,
    Validator,
    UniversalJSONEncoder,
    JSONFormatter,
    configure_logging,
    PhonemeCategory
)

# =============================================================================
# 1. Enhanced Logging Configuration for Demo
# =============================================================================

def initialize_demo_logging():
    """
    Configure logging specifically for this demo.
    Logs are emitted in JSON format, which can be displayed neatly with the Rich library.
    """
    LOG_LEVEL = os.getenv("UNIVERSAL_COMPONENTS_DEMO_LOG_LEVEL", "DEBUG")
    LOG_FILE = os.getenv("UNIVERSAL_COMPONENTS_DEMO_LOG_FILE", "universal_components_demo.log")
    configure_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)
    logging.info(f"Logging initialized with level {LOG_LEVEL} and log file '{LOG_FILE}'.")

# =============================================================================
# 2. Validator Execution
# =============================================================================

def run_validator():
    """
    Executes the Validator to ensure all components in universal_components.py are valid.
    Logs any errors or warnings for immediate review.
    """
    logging.info("Running Validator to check the integrity of universal_components...")
    validator = Validator()
    validator.run_all_validations()
    validator.report()
    if validator.errors:
        logging.error("Validator detected errors. Please review the log for details.")
    else:
        logging.info("Validator passed without any critical issues.")

# =============================================================================
# 3. Listing Components
# =============================================================================

def list_computational_instructions():
    """
    Lists all computational instructions with their details (description, operands, examples).
    """
    instructions = ComputationalInstructions.list_all_instructions()
    logging.info("=== Listing All Computational Instructions ===")
    for instr in instructions:
        logging.info(f"- {instr.name.name}: {instr.description}")
        logging.debug(f"  Operands: {instr.operands}")
        logging.debug(f"  Example Usage: {instr.example_usage}")

def list_mathematical_functions():
    """
    Lists all available mathematical functions by category, displaying their properties.
    """
    logging.info("=== Listing All Mathematical Functions ===")
    for category in MathematicalFunctionCategory:
        funcs = MathematicalFunctions.get_functions_by_category(category)
        logging.info(f"\nCategory: {category.name}")
        for func in funcs:
            logging.info(f"  - {func.name}: {func.description}")
            logging.debug(f"    Symbol: {func.symbol}")
            logging.debug(f"    Domain: {func.domain}")
            logging.debug(f"    Range: {func.range}")
            logging.debug(f"    Properties: {func.properties}")
            logging.debug(f"    Related Functions: {func.related_functions}")

def list_logic_symbols():
    """
    Lists all irreducible logic symbols, including their precedence and associativity.
    """
    symbols = IrreducibleLogicSymbols.list_all_logic_symbols()
    logging.info("=== Listing All Logic Symbols ===")
    for symbol in symbols:
        logging.info(f"- {symbol.operator.name}: {symbol.symbol}")
        logging.debug(f"  Description: {symbol.description}")
        logging.debug(f"  Precedence: {symbol.precedence}")
        logging.debug(f"  Associativity: {symbol.associativity}")

def list_phonemes():
    """
    Lists all English phonemes with relevant details, such as examples and IPA notation.
    """
    phonemes = IrreducibleEnglishPhonemes.list_all_phonemes()
    logging.info("=== Listing All English Phonemes ===")
    for phoneme in phonemes:
        logging.info(f"- {phoneme.symbol}: {phoneme.description}")
        logging.debug(f"  Example: {phoneme.example}")
        logging.debug(f"  Category: {phoneme.category}")
        logging.debug(f"  IPA: {phoneme.ipa}")
        logging.debug(f"  Example Sentence: {phoneme.example_sentence}")

# =============================================================================
# 4. Searching Components
# =============================================================================

def search_computational_instructions(keyword: str):
    """
    Searches computational instructions by keyword in name or description.
    """
    results = ComputationalInstructions.search_instructions(keyword)
    logging.info(f"=== Search Results for Computational Instructions containing '{keyword}' ===")
    for instr in results:
        logging.info(f"- {instr.name.name}: {instr.description}")

def search_mathematical_functions(keyword: str):
    """
    Searches mathematical functions by keyword in name or description.
    """
    # We'll add a custom search method to MathematicalFunctions if not already present
    def _search_math_funcs(k: str) -> List[MathematicalFunction]:
        matches = []
        for cat_funcs in MathematicalFunctions.functions.values():
            for f in cat_funcs:
                if k.lower() in f.name.lower() or k.lower() in f.description.lower():
                    matches.append(f)
        return matches

    results = _search_math_funcs(keyword)
    logging.info(f"=== Search Results for Mathematical Functions containing '{keyword}' ===")
    for func in results:
        logging.info(f"- {func.name}: {func.description}")

def search_logic_symbols(keyword: str):
    """
    Searches logic symbols by keyword in operator name or description.
    """
    results = IrreducibleLogicSymbols.search_logic_symbols(keyword)
    logging.info(f"=== Search Results for Logic Symbols containing '{keyword}' ===")
    for symbol in results:
        logging.info(f"- {symbol.operator.name}: {symbol.symbol} - {symbol.description}")

def search_phonemes(keyword: str):
    """
    Searches phonemes by keyword in symbol or description.
    """
    results = IrreducibleEnglishPhonemes.search_phonemes(keyword)
    logging.info(f"=== Search Results for Phonemes containing '{keyword}' ===")
    for phoneme in results:
        logging.info(f"- {phoneme.symbol}: {phoneme.description}")

# =============================================================================
# 5. Adding Components
# =============================================================================

def add_new_computational_instruction():
    """
    Adds a new computational instruction if it doesn't already exist.
    Demonstrates adding an entry to the global dictionary in ComputationalInstructions.
    """
    if ComputationalInstruction.DEFINE_CONSTANT in ComputationalInstructions.instructions:
        logging.warning(f"Instruction '{ComputationalInstruction.DEFINE_CONSTANT.name}' already exists. Skipping addition.")
        return

    new_instr = InstructionDetail(
        name=ComputationalInstruction.DEFINE_CONSTANT,
        description="Define a constant value.",
        operands=["constant_label", "value"],
        example_usage="DEFINE_CONSTANT PI, 3.14159"
    )
    try:
        ComputationalInstructions.add_instruction(new_instr)
        logging.info(f"Successfully added new instruction: {new_instr.name.name}")
    except ValueError as e:
        logging.error(f"Error adding instruction: {e}")

def add_new_mathematical_function():
    """
    Adds a new mathematical function to the EXPONENTIAL_LOGARITHMIC category.
    Demonstrates usage of add_function.
    """
    new_func = MathematicalFunction(
        name="Logistic Function",
        symbol="f(x) = 1 / (1 + e^(-x))",
        category=MathematicalFunctionCategory.EXPONENTIAL_LOGARITHMIC,
        description="A sigmoid function used in statistics and machine learning.",
        domain="All real numbers",
        range="(0, 1)",
        properties=["Continuous", "S-shaped curve"],
        related_functions=["Exponential Function", "Hyperbolic Tangent"]
    )
    try:
        MathematicalFunctions.add_function(new_func)
        logging.info(f"Successfully added new mathematical function: {new_func.name}")
    except ValueError as e:
        logging.error(f"Error adding mathematical function: {e}")

def add_new_logic_symbol():
    """
    Adds a new logic symbol if it doesn't already exist in IrreducibleLogicSymbols.
    """
    if any(symbol.operator == LogicOperator.BICONDITIONAL for symbol in IrreducibleLogicSymbols.symbols):
        logging.warning(f"Logic symbol '{LogicOperator.BICONDITIONAL.name}' already exists. Skipping addition.")
        return

    new_symbol = LogicSymbol(
        operator=LogicOperator.BICONDITIONAL,
        symbol="↔",
        description="Logical biconditional, representing 'if and only if'.",
        precedence=1,
        associativity="Left"
    )
    try:
        IrreducibleLogicSymbols.add_logic_symbol(new_symbol)
        logging.info(f"Successfully added new logic symbol: {new_symbol.operator.name}")
    except ValueError as e:
        logging.error(f"Error adding logic symbol: {e}")

def add_new_phoneme():
    """
    Adds a new phoneme symbol /θr/ if it doesn't already exist in the phoneme list.
    """
    if any(p.symbol == "/θr/" for p in IrreducibleEnglishPhonemes._phonemes):
        logging.warning("Phoneme '/θr/' already exists. Skipping addition.")
        return

    new_phoneme = Phoneme(
        symbol="/θr/",
        example='"three"',
        category=PhonemeCategory.CONSONANT,
        description="Voiceless dental fricative followed by 'r', as in 'three'.",
        ipa="θr",
        example_sentence='"She has three apples."'
    )
    try:
        IrreducibleEnglishPhonemes.add_phoneme(new_phoneme)
        logging.info(f"Successfully added new phoneme: {new_phoneme.symbol}")
    except ValueError as e:
        logging.error(f"Error adding phoneme: {e}")

# =============================================================================
# 6. Modifying Components
# =============================================================================

def modify_computational_instruction():
    """
    Modifies an existing computational instruction (ADD) to include an updated description and usage.
    """
    try:
        ComputationalInstructions.modify_instruction(
            ComputationalInstruction.ADD,
            description="Add two numbers and store the result in a destination register.",
            operands=["destination_register", "source_register1", "source_register2"],
            example_usage="ADD R3, R1, R2"
        )
        logging.info("Successfully modified instruction: ADD")
    except KeyError as e:
        logging.error(f"Error modifying instruction: {e}")

def modify_mathematical_function():
    """
    Modifies 'Linear Function' in the POLYNOMIAL category to extend its description and properties.
    """
    # We'll add a custom function if not present in MathematicalFunctions
    def modify_func(category: MathematicalFunctionCategory, function_name: str, **kwargs):
        cat_funcs = MathematicalFunctions.functions.get(category, [])
        for idx, f in enumerate(cat_funcs):
            if f.name.lower() == function_name.lower():
                # Create an updated version of the function
                updated = MathematicalFunction(
                    name=kwargs.get("name", f.name),
                    symbol=kwargs.get("symbol", f.symbol),
                    category=kwargs.get("category", f.category),
                    description=kwargs.get("description", f.description),
                    domain=kwargs.get("domain", f.domain),
                    range=kwargs.get("range", f.range),
                    properties=kwargs.get("properties", f.properties),
                    related_functions=kwargs.get("related_functions", f.related_functions)
                )
                cat_funcs[idx] = updated
                return
        raise KeyError(f"Mathematical Function '{function_name}' not found in category '{category.name}'.")

    try:
        modify_func(
            MathematicalFunctionCategory.POLYNOMIAL,
            "Linear Function",
            description="A first-degree polynomial function: f(x) = mx + b, representing a straight line.",
            properties=["Linear", "Degree 1", "One-to-one", "Continuous"]
        )
        logging.info("Successfully modified mathematical function: Linear Function")
    except KeyError as e:
        logging.error(f"Error modifying mathematical function: {e}")

def modify_logic_symbol():
    """
    Modifies an existing logic symbol (IMPLICATION) to give it a different association or precedence.
    """
    try:
        IrreducibleLogicSymbols.modify_logic_symbol(
            LogicOperator.IMPLICATION,
            description="Logical implication: 'if...then...'. Updated for demonstration.",
            precedence=1,
            associativity="Right"
        )
        logging.info("Successfully modified logic symbol: IMPLICATION")
    except (KeyError, ValueError) as e:
        logging.error(f"Error modifying logic symbol: {e}")

def modify_phoneme():
    """
    Modifies an existing phoneme (/p/) to update its description or example sentence.
    """
    try:
        IrreducibleEnglishPhonemes.modify_phoneme(
            symbol="/p/",
            description="Voiceless bilabial plosive as in 'pat'. New updated description with more detail.",
            example_sentence='"Pat placed the pen."'
        )
        logging.info("Successfully modified phoneme: /p/")
    except KeyError as e:
        logging.error(f"Error modifying phoneme: {e}")

# =============================================================================
# 7. Removing Components
# =============================================================================

def remove_computational_instruction():
    """
    Removes an existing computational instruction (NOP) if present.
    """
    try:
        ComputationalInstructions.remove_instruction(ComputationalInstruction.NOP)
        logging.info("Successfully removed instruction: NOP")
    except KeyError as e:
        logging.error(f"Error removing instruction: {e}")

def remove_mathematical_function():
    """
    Removes 'Max Function' from MISCELLANEOUS category if it exists.
    """
    # We'll add a custom remove function to MathematicalFunctions (similar to remove_function).
    def remove_func(category: MathematicalFunctionCategory, function_name: str):
        cat_funcs = MathematicalFunctions.functions.get(category, [])
        for f in cat_funcs:
            if f.name.lower() == function_name.lower():
                cat_funcs.remove(f)
                return
        raise KeyError(f"Mathematical Function '{function_name}' not found in category '{category.name}'.")

    try:
        remove_func(MathematicalFunctionCategory.MISCELLANEOUS, "Max Function")
        logging.info("Successfully removed mathematical function: Max Function")
    except KeyError as e:
        logging.error(f"Error removing mathematical function: {e}")

def remove_logic_symbol():
    """
    Removes the XOR logic symbol if it exists.
    """
    try:
        IrreducibleLogicSymbols.remove_logic_symbol(LogicOperator.XOR)
        logging.info("Successfully removed logic symbol: XOR")
    except KeyError as e:
        logging.error(f"Error removing logic symbol: {e}")

def remove_phoneme():
    """
    Removes the /w/ phoneme if present.
    """
    try:
        IrreducibleEnglishPhonemes.remove_phoneme("/w/")
        logging.info("Successfully removed phoneme: /w/")
    except KeyError as e:
        logging.error(f"Error removing phoneme: {e}")

# =============================================================================
# 8. Serialization and Deserialization
# =============================================================================

def serialize_components():
    """
    Serializes each major component group (instructions, functions, symbols, phonemes) to JSON.
    """
    try:
        ci_json = ComputationalInstructions.to_json()
        with open("computational_instructions.json", "w") as f:
            f.write(ci_json)
        logging.info("Serialized Computational Instructions to 'computational_instructions.json'.")

        mf_json = MathematicalFunctions.to_json()
        with open("mathematical_functions.json", "w") as f:
            f.write(mf_json)
        logging.info("Serialized Mathematical Functions to 'mathematical_functions.json'.")

        ls_json = IrreducibleLogicSymbols.to_json()
        with open("logic_symbols.json", "w") as f:
            f.write(ls_json)
        logging.info("Serialized Logic Symbols to 'logic_symbols.json'.")

        ep_json = IrreducibleEnglishPhonemes.to_json()
        with open("english_phonemes.json", "w") as f:
            f.write(ep_json)
        logging.info("Serialized English Phonemes to 'english_phonemes.json'.")
    except Exception as e:
        logging.error(f"Error during serialization: {e}")

def deserialize_components():
    """
    Deserializes components from the JSON files if they exist in the current directory.
    """
    try:
        if os.path.exists("computational_instructions.json"):
            with open("computational_instructions.json", "r") as f:
                ci_json = f.read()
            ComputationalInstructions.from_json(ci_json)
            logging.info("Deserialized Computational Instructions from 'computational_instructions.json'.")

        if os.path.exists("mathematical_functions.json"):
            with open("mathematical_functions.json", "r") as f:
                mf_json = f.read()
            MathematicalFunctions.from_json(mf_json)
            logging.info("Deserialized Mathematical Functions from 'mathematical_functions.json'.")

        if os.path.exists("logic_symbols.json"):
            with open("logic_symbols.json", "r") as f:
                ls_json = f.read()
            IrreducibleLogicSymbols.from_json(ls_json)
            logging.info("Deserialized Logic Symbols from 'logic_symbols.json'.")

        if os.path.exists("english_phonemes.json"):
            with open("english_phonemes.json", "r") as f:
                ep_json = f.read()
            IrreducibleEnglishPhonemes.from_json(ep_json)
            logging.info("Deserialized English Phonemes from 'english_phonemes.json'.")
    except Exception as e:
        logging.error(f"Error during deserialization: {e}")

# =============================================================================
# 9. Simulated Usage
# =============================================================================

class SimpleVirtualMachine:
    """
    A simple virtual machine that executes a list of computational instructions.
    Demonstrates how instructions can be interpreted in a real scenario.
    """

    def __init__(self):
        self.registers: Dict[str, float] = {}
        self.memory: Dict[str, float] = {}
        self.stack: List[float] = []
        self.program_counter: int = 0
        self.instructions: List[InstructionDetail] = []
        self.running: bool = True

    def load_instructions(self, instructions: List[InstructionDetail]):
        """
        Loads instructions into the VM for sequential execution.
        """
        self.instructions = instructions
        logging.info(f"Loaded {len(instructions)} instructions into the Virtual Machine.")

    def execute_next(self):
        """
        Fetches and executes the next instruction in the program.
        """
        if self.program_counter >= len(self.instructions):
            logging.info("Program counter has reached the end of instructions.")
            self.running = False
            return

        instr_detail = self.instructions[self.program_counter]
        logging.debug(f"Executing Instruction: {instr_detail.example_usage}")

        # Simple approach: parse example_usage to identify the instruction and operands
        parts = instr_detail.example_usage.split(',')
        instr_and_operands = parts[0].split()
        instr_name = instr_and_operands[0]
        operands = (
            [op.strip() for op in instr_and_operands[1:]] +
            [op.strip() for op in parts[1:]] if len(parts) > 1 else
            instr_and_operands[1:]
        )

        try:
            if instr_name == "LOAD":
                reg, mem_addr = operands
                self.registers[reg] = self.memory.get(mem_addr, 0.0)
                logging.debug(f"Loaded {self.registers[reg]} into {reg} from {mem_addr}.")
            elif instr_name == "STORE":
                reg, mem_addr = operands
                self.memory[mem_addr] = self.registers.get(reg, 0.0)
                logging.debug(f"Stored {self.memory[mem_addr]} from {reg} into {mem_addr}.")
            elif instr_name == "ADD":
                dest, src1, src2 = operands
                self.registers[dest] = self.registers.get(src1, 0.0) + self.registers.get(src2, 0.0)
                logging.debug(f"Added {src1} and {src2}, result {self.registers[dest]} stored in {dest}.")
            elif instr_name == "SUB":
                dest, src1, src2 = operands
                self.registers[dest] = self.registers.get(src1, 0.0) - self.registers.get(src2, 0.0)
                logging.debug(f"Subtracted {src2} from {src1}; result {self.registers[dest]} stored in {dest}.")
            elif instr_name == "PUSH":
                reg_or_val = operands[0]
                try:
                    val = float(reg_or_val.strip('"'))
                except ValueError:
                    val = self.registers.get(reg_or_val, 0.0)
                self.stack.append(val)
                logging.debug(f"Pushed {val} onto the stack.")
            elif instr_name == "POP":
                reg = operands[0]
                if self.stack:
                    self.registers[reg] = self.stack.pop()
                    logging.debug(f"Popped value {self.registers[reg]} into {reg}.")
                else:
                    logging.warning("Stack is empty. POP operation skipped.")
            elif instr_name == "PRINT":
                reg_or_val = operands[0]
                output_val = self.registers.get(reg_or_val, reg_or_val.strip('"'))
                logging.info(f"PRINT: {output_val}")
            elif instr_name == "HALT":
                logging.info("HALT instruction encountered. Stopping execution.")
                self.running = False
            elif instr_name == "DEFINE_CONSTANT":
                const_label, const_value = operands
                try:
                    self.memory[const_label] = float(const_value)
                    logging.debug(f"Defined constant {const_label} with value {const_value}.")
                except ValueError:
                    logging.error(f"Invalid value for DEFINE_CONSTANT: {const_value}")
            else:
                logging.warning(f"Unknown instruction '{instr_name}' encountered.")
        except IndexError:
            logging.error(f"Insufficient operands for instruction '{instr_name}'.")
        except ValueError as ve:
            logging.error(f"ValueError: {ve} in instruction '{instr_name}'.")

        self.program_counter += 1

    def run(self):
        """
        Continues executing instructions until HALT or end-of-instructions.
        """
        logging.info("Starting Virtual Machine Execution.")
        while self.running:
            self.execute_next()
        logging.info("Virtual Machine Execution Completed.")

def simulate_virtual_machine():
    """
    Demonstrates using the SimpleVirtualMachine to run a small program of instructions.
    """
    vm = SimpleVirtualMachine()
    program = [
        ComputationalInstructions.get_instruction_detail(ComputationalInstruction.DEFINE_CONSTANT),
        ComputationalInstructions.get_instruction_detail(ComputationalInstruction.LOAD),
        ComputationalInstructions.get_instruction_detail(ComputationalInstruction.ADD),
        ComputationalInstructions.get_instruction_detail(ComputationalInstruction.PUSH),
        ComputationalInstructions.get_instruction_detail(ComputationalInstruction.POP),
        ComputationalInstructions.get_instruction_detail(ComputationalInstruction.PRINT),
        ComputationalInstructions.get_instruction_detail(ComputationalInstruction.HALT)
    ]
    # Mock memory and registers
    vm.memory = {"PI": 3.14159, "0x1000": 10.0, "0x1004": 20.0}
    vm.registers = {"R1": 5.0, "R2": 15.0}

    vm.load_instructions(program)
    vm.run()

# =============================================================================
# 10. Additional Complex Demonstrations
# =============================================================================

# 10.1 Logical Expressions
def evaluate_logical_expression(expression: str) -> bool:
    """
    Evaluates a logical expression that uses tokens: True, False, AND, OR, NOT, XOR, IMPLICATION, BICONDITIONAL.
    """
    ops = {
        'AND': operator.and_,
        'OR': operator.or_,
        'NOT': operator.not_,
        'XOR': lambda a, b: a ^ b,
        'IMPLICATION': lambda a, b: (not a) or b,
        'BICONDITIONAL': operator.eq
    }

    tokens = expression.split()
    stack = []
    current_op = None

    try:
        for token in tokens:
            if token in ops:
                # If the token is 'NOT', handle it as a unary operator
                if token == 'NOT':
                    current_op = ops[token]
                    continue
                else:
                    current_op = ops[token]
            elif token in ['True', 'False']:
                val = (token == 'True')
                # If current_op is a unary operator NOT
                if current_op == operator.not_:
                    neg_val = current_op(val)
                    stack.append(neg_val)
                    current_op = None
                elif current_op and stack:
                    left_val = stack.pop()
                    result = current_op(left_val, val)
                    stack.append(result)
                    current_op = None
                else:
                    stack.append(val)
            else:
                logging.error(f"Invalid token '{token}' found in expression.")
                return False
        return stack.pop() if stack else False
    except IndexError:
        logging.error("Insufficient operands while evaluating expression.")
        return False
    except Exception as e:
        logging.error(f"Error while evaluating expression '{expression}': {e}")
        return False

def demonstrate_logic_symbols():
    """
    Demonstrates usage of logic operators by evaluating a series of example expressions.
    """
    expressions = [
        "True AND False",
        "True OR False",
        "NOT True",
        "True XOR False",
        "True IMPLICATION False",
        "True BICONDITIONAL True",
        "False IMPLICATION True",
        "False BICONDITIONAL True"
    ]
    logging.info("=== Demonstrating Logical Expression Evaluations ===")
    for expr in expressions:
        result = evaluate_logical_expression(expr)
        logging.info(f"Expression: '{expr}' => Result: {result}")

# 10.2 Mathematical Function Computations
def compute_mathematical_functions():
    """
    Iterates over a selection of known math functions (like Sine, Cosine, etc.) and computes them on sample values.
    Highlights how these definitions can be leveraged in real computations.
    """
    import math

    logging.info("=== Computing Various Mathematical Functions ===")
    all_funcs = []
    for cat_funcs in MathematicalFunctions.functions.values():
        all_funcs.extend(cat_funcs)

    # Remove duplicates if any (by name)
    seen = set()
    unique_funcs = []
    for f in all_funcs:
        if f.name not in seen:
            unique_funcs.append(f)
            seen.add(f.name)

    sample_x_values = [0, 1, -1, math.pi/2, 2*math.pi]

    for func in unique_funcs:
        # Basic examples (not exhaustive)
        if "sine" in func.name.lower():
            for x in sample_x_values:
                val = round(math.sin(x), 6)
                logging.info(f"{func.name}({round(x,2)}) = {val}")
        elif "cosine" in func.name.lower():
            for x in sample_x_values:
                val = round(math.cos(x), 6)
                logging.info(f"{func.name}({round(x,2)}) = {val}")
        elif "tangent" in func.name.lower():
            for x in sample_x_values:
                try:
                    val = round(math.tan(x), 6)
                except ValueError:
                    val = "Undefined"
                logging.info(f"{func.name}({round(x,2)}) = {val}")
        elif "exponential" in func.name.lower():
            for x in sample_x_values:
                val = round(math.exp(x), 6)
                logging.info(f"{func.name}({round(x,2)}) = {val}")
        elif "logarithm" in func.name.lower():
            for x in sample_x_values:
                if x > 0:
                    # Distinguish natural vs. common vs. binary logs if we like
                    val = round(math.log(x), 6)
                else:
                    val = "Undefined"
                logging.info(f"{func.name}({round(x,2)}) = {val}")
        else:
            logging.debug(f"Skipping advanced or special function: {func.name}")

# 10.3 Phoneme Processing
def get_phoneme_by_symbol(symbol: str) -> Optional[Phoneme]:
    """
    Retrieves a phoneme by its symbol from IrreducibleEnglishPhonemes.
    """
    for p in IrreducibleEnglishPhonemes._phonemes:
        if p.symbol == symbol:
            return p
    return None

def process_phonemes():
    """
    Processes a sample text, mapping each character to a phoneme (if defined),
    and logs the findings.
    """
    sample_text = "Hello, world! This is a comprehensive demonstration of universal components."
    logging.info(f"=== Processing Phonemes for Sample Text: '{sample_text}' ===")

    char_to_phoneme = {
        'h': "/h/",
        'e': "/ɛ/",
        'l': "/l/",
        'o': "/oʊ/",
        'w': "/w/",
        'r': "/r/",
        'd': "/d/",
        't': "/t/",
        'i': "/ɪ/",
        's': "/s/",
        'a': "/æ/",
        'c': "/k/",
        'm': "/m/",
        'p': "/p/",
        'n': "/n/",
        'g': "/g/",
        ' ': " ",
        ',': ",",
        '!': "!",
        '.': ".",
        'f': "/f/",
        'u': "/ʊ/",
        'b': "/b/",
        'y': "/j/",
        'k': "/k/",
        'v': "/v/",
        'z': "/z/",
        'q': "/kw/",
        'j': "/dʒ/",
        'x': "/ks/"
        # Could add more as needed
    }

    for char in sample_text.lower():
        phoneme_symbol = char_to_phoneme.get(char, None)
        if phoneme_symbol and phoneme_symbol not in [" ", ",", "!", "."]:
            p_detail = get_phoneme_by_symbol(phoneme_symbol)
            if p_detail:
                logging.info(f"Character '{char}' => Phoneme '{p_detail.symbol}': {p_detail.description}")
            else:
                logging.warning(f"No detail found for phoneme '{phoneme_symbol}' mapped from '{char}'.")
        elif phoneme_symbol in [" ", ",", "!", "."]:
            logging.debug(f"Character '{char}' is punctuation or space.")
        else:
            logging.warning(f"No phoneme mapping found for character: '{char}'")

# 10.4 Natural Language to Computational Instructions
def nl_to_computational_instructions():
    """
    Demonstrates regex-based parsing of natural language commands to produce instructions.
    Shows how user input might be dynamically converted to code instructions.
    """
    import re

    commands = [
        "Load the value from memory address 0x1000 into register R1.",
        "Add the values in R1 and R2 and store the result in R3.",
        "Push the value of R3 onto the stack.",
        "Pop the top value from the stack into register R4.",
        "Print the value of R4."
    ]

    logging.info("=== Converting Natural Language Commands to Computational Instructions ===")

    # Regex patterns to parse instructions
    load_pattern = r"Load the value from memory address (?P<mem_addr>0x[0-9A-Fa-f]+) into register (?P<reg>R\d+)\."
    store_pattern = r"Store the value from register (?P<reg>R\d+) into memory address (?P<mem_addr>0x[0-9A-Fa-f]+)\."
    add_pattern = r"Add the values in (?P<src1>R\d+) and (?P<src2>R\d+) and store the result in (?P<dest>R\d+)\."
    sub_pattern = r"Subtract the value in register (?P<src2>R\d+) from register (?P<src1>R\d+) and store the result in (?P<dest>R\d+)\."
    push_pattern = r"Push the value of (?P<reg>R\d+) onto the stack\."
    pop_pattern = r"Pop the top value from the stack into register (?P<reg>R\d+)\."
    print_pattern = r"Print the value of (?P<reg>R\d+)\."

    for command in commands:
        instr_enum = None
        operands = []
        example_usage = ""

        if re.match(load_pattern, command):
            m = re.match(load_pattern, command)
            mem_addr = m.group("mem_addr")
            reg = m.group("reg")
            instr_enum = ComputationalInstruction.LOAD
            operands = [reg, mem_addr]
            example_usage = f"LOAD {reg}, {mem_addr}"
        elif re.match(store_pattern, command):
            m = re.match(store_pattern, command)
            reg = m.group("reg")
            mem_addr = m.group("mem_addr")
            instr_enum = ComputationalInstruction.STORE
            operands = [reg, mem_addr]
            example_usage = f"STORE {reg}, {mem_addr}"
        elif re.match(add_pattern, command):
            m = re.match(add_pattern, command)
            src1 = m.group("src1")
            src2 = m.group("src2")
            dest = m.group("dest")
            instr_enum = ComputationalInstruction.ADD
            operands = [dest, src1, src2]
            example_usage = f"ADD {dest}, {src1}, {src2}"
        elif re.match(sub_pattern, command):
            m = re.match(sub_pattern, command)
            src1 = m.group("src1")
            src2 = m.group("src2")
            dest = m.group("dest")
            instr_enum = ComputationalInstruction.SUB
            operands = [dest, src1, src2]
            example_usage = f"SUB {dest}, {src1}, {src2}"
        elif re.match(push_pattern, command):
            m = re.match(push_pattern, command)
            reg = m.group("reg")
            instr_enum = ComputationalInstruction.PUSH
            operands = [reg]
            example_usage = f"PUSH {reg}"
        elif re.match(pop_pattern, command):
            m = re.match(pop_pattern, command)
            reg = m.group("reg")
            instr_enum = ComputationalInstruction.POP
            operands = [reg]
            example_usage = f"POP {reg}"
        elif re.match(print_pattern, command):
            m = re.match(print_pattern, command)
            reg = m.group("reg")
            instr_enum = ComputationalInstruction.PRINT
            operands = [reg]
            example_usage = f"PRINT {reg}"
        else:
            logging.warning(f"Unrecognized command format: '{command}'")
            continue

        if instr_enum:
            detail = InstructionDetail(
                name=instr_enum,
                description=f"Auto-generated from NLP command: '{command}'",
                operands=operands,
                example_usage=example_usage
            )
            try:
                ComputationalInstructions.add_instruction(detail)
                logging.info(f"Added Instruction from NLP: {detail.example_usage}")
            except ValueError:
                logging.warning(f"Instruction '{instr_enum.name}' already exists. Skipping.")

# 10.5 PyTorch-Based Function Approximation
def pytorch_function_approximation():
    """
    Uses PyTorch to approximate the sine function using a small neural network.
    Demonstrates how universal_components can integrate with ML frameworks.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import math

    class SineApproximator(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(1, 50)
            self.activation = nn.Tanh()
            self.output = nn.Linear(50, 1)

        def forward(self, x):
            x = self.activation(self.hidden(x))
            return self.output(x)

    def generate_data(num_samples=1000):
        x = torch.linspace(-math.pi, math.pi, num_samples).unsqueeze(1)
        y = torch.sin(x)
        return x, y

    model = SineApproximator()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    x_train, y_train = generate_data()

    def train_model(epochs=1000):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(x_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 200 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.6f}")

    logging.info("=== PyTorch Sine Function Approximation ===")
    train_model()

    model.eval()
    with torch.no_grad():
        test_samples = torch.tensor([[0.0], [1.57], [-1.57], [3.14], [-3.14]])
        preds = model(test_samples)
        for inp, pred in zip(test_samples, preds):
            real_val = math.sin(inp.item())
            logging.info(f"Input: {inp.item():.2f}, Prediction: {pred.item():.6f}, Actual Sine: {real_val:.6f}")

# 10.6 Logical Sentiment Analysis
def logical_sentiment_analysis():
    """
    Performs sentiment analysis on sample text using TextBlob,
    then applies basic logic to classify the sentiment as Positive, Negative, or Neutral.
    """

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    samples = [
        "I absolutely love this product!",
        "This is the worst thing I've ever bought!",
        "It's okay, I guess.",
        "Not sure if I like it or hate it.",
        "Fantastic experience overall."
    ]

    logging.info("=== Performing Logical Sentiment Analysis ===")
    for text in samples:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        is_positive = polarity > 0
        is_negative = polarity < 0
        is_neutral = abs(polarity) < 0.05

        if is_positive and not is_negative:
            overall = "Positive"
        elif is_negative and not is_positive:
            overall = "Negative"
        else:
            overall = "Neutral"

        logging.info(f"Text: '{text}'")
        logging.info(f"  Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")
        logging.info(f"  Overall Sentiment: {overall}")

# 10.7 Symbolic Equation Solver
def symbolic_equation_solver():
    """
    Uses sympy to solve basic algebraic equations symbolically.
    """
    import sympy
    x = sympy.Symbol('x', real=True)
    eqs = [
        sympy.Eq(2*x + 3, 7),
        sympy.Eq(x**2 - 4, 0),
        sympy.Eq(sympy.sin(x), 0),
        sympy.Eq(sympy.exp(x), 1),
        sympy.Eq(x/2 - 5, 0)
    ]

    logging.info("=== Solving Symbolic Equations ===")
    for eq in eqs:
        solution = sympy.solve(eq, x)
        logging.info(f"Equation: {eq}")
        logging.info(f"  Solutions: {solution}")

# 10.8 Phoneme-Based Text Generation
def phoneme_based_text_generation():
    """
    An LSTM-based RNN example that "learns" simple phoneme sequences and can generate new sequences.
    This is purely demonstrative and not intended for production usage.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    # Gather phonemes from IrreducibleEnglishPhonemes
    phonemes = [p.symbol for p in IrreducibleEnglishPhonemes._phonemes]
    idx_map = {ph: i for i, ph in enumerate(phonemes)}
    rev_idx_map = {v: k for k, v in idx_map.items()}

    class PhonemeRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x, hidden):
            out, hidden = self.lstm(x, hidden)
            out = self.fc(out)
            return out, hidden

        def init_hidden(self):
            return (torch.zeros(self.num_layers, 1, self.hidden_size),
                    torch.zeros(self.num_layers, 1, self.hidden_size))

    input_size = len(phonemes)
    hidden_size = 128
    output_size = len(phonemes)
    model = PhonemeRNN(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Construct a small dataset of phoneme sequences
    sample_sequence = [idx_map.get("/h/"), idx_map.get("/ɛ/"), idx_map.get("/l/"), idx_map.get("/oʊ/")]
    # Filter out None
    sample_sequence = [s for s in sample_sequence if s is not None]

    if len(sample_sequence) < 2:
        logging.error("Not enough phonemes to build a sample sequence. Please ensure '/h/', '/ɛ/', '/l/', '/oʊ/' exist.")
        return

    # Build repeated sequences
    data_sequences = []
    for _ in range(200):  # repeat to build a 'larger' training set
        data_sequences.append(sample_sequence)

    # Prepare training data
    inputs_list = []
    targets_list = []
    for seq in data_sequences:
        inp = seq[:-1]
        tgt = seq[1:]
        inputs_list.append(inp)
        targets_list.append(tgt)

    # One-hot inputs
    inputs_tensor = torch.zeros(len(inputs_list), len(inputs_list[0]), input_size)
    for i, seq in enumerate(inputs_list):
        for t, idx_val in enumerate(seq):
            inputs_tensor[i, t, idx_val] = 1.0

    targets_tensor = torch.tensor(targets_list)

    # Training
    def train_rnn(epochs=800):
        for epoch in range(epochs):
            hidden = model.init_hidden()
            optimizer.zero_grad()
            outputs, hidden = model(inputs_tensor, hidden)
            loss = criterion(outputs.view(-1, output_size), targets_tensor.view(-1))
            loss.backward()
            optimizer.step()

            if (epoch+1) % 200 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    logging.info("=== Starting Phoneme-Based Text Generation Training ===")
    train_rnn()

    # Generation
    def generate_phoneme_seq(start_idx, length=5):
        model.eval()
        hidden = model.init_hidden()
        input_vec = torch.zeros(1, 1, input_size)
        input_vec[0, 0, start_idx] = 1.0

        generated = [rev_idx_map.get(start_idx, "")]
        with torch.no_grad():
            for _ in range(length):
                out, hidden = model(input_vec, hidden)
                probs = torch.softmax(out[0, -1], dim=0).numpy()
                next_idx = np.random.choice(range(len(phonemes)), p=probs)
                generated.append(rev_idx_map.get(next_idx, ""))
                input_vec = torch.zeros(1, 1, input_size)
                input_vec[0, 0, next_idx] = 1.0
        return generated

    if sample_sequence:
        gen_seq = generate_phoneme_seq(sample_sequence[0])
        logging.info(f"Generated Phoneme Sequence: {' '.join(gen_seq)}")

# =============================================================================
# 11. Additional Wrapper for Enhanced Log Display (Optional)
# =============================================================================

def process_log_file():
    """
    Reads the demo log file and presents it in a styled Rich table if Rich is installed.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        logging.error("Rich is not installed. Skipping log display.")
        return

    LOG_FILE = os.getenv("UNIVERSAL_COMPONENTS_DEMO_LOG_FILE", "universal_components_demo.log")
    if not os.path.exists(LOG_FILE):
        logging.error(f"Log file '{LOG_FILE}' does not exist. Cannot process logs.")
        return

    console = Console()
    table = Table(title="Universal Components Demo Log", show_lines=True)
    table.add_column("Timestamp", style="cyan", no_wrap=True)
    table.add_column("Level", style="magenta", no_wrap=True)
    table.add_column("Message", style="white")
    table.add_column("Module", style="green")
    table.add_column("Function", style="yellow")
    table.add_column("Line", style="blue", justify="right")

    level_styles = {
        "DEBUG": "dim",
        "INFO": "white",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red"
    }

    with open(LOG_FILE, "r") as logfile:
        for line in logfile:
            try:
                record = json.loads(line)
                ts = record.get("timestamp", "")
                lvl = record.get("level", "").upper()
                msg = record.get("message", "")
                mod = record.get("module", "")
                fn = record.get("funcName", "")
                ln = str(record.get("lineNo", ""))

                lvl_style = level_styles.get(lvl, "white")
                lvl_text = Text(lvl, style=lvl_style)
                msg_text = Text(msg, style="white")
                mod_text = Text(mod, style="green")
                fn_text = Text(fn, style="yellow")
                ln_text = Text(ln, style="blue")

                table.add_row(ts, lvl_text, msg_text, mod_text, fn_text, ln_text)
            except json.JSONDecodeError:
                continue

    console.print(table)

# =============================================================================
# 12. Main Demo Function
# =============================================================================

def main():
    """
    Main function to orchestrate the entire universal_components demo.
    """
    initialize_demo_logging()
    logging.info("=== Universal Components Demo Started ===\n")

    # 1. Run Validation
    run_validator()

    # 2. List all components
    list_computational_instructions()
    list_mathematical_functions()
    list_logic_symbols()
    list_phonemes()

    # 3. Search in components
    search_computational_instructions("Load")
    search_mathematical_functions("Function")
    search_logic_symbols("AND")
    search_phonemes("Vowel")

    # 4. Add new components
    add_new_computational_instruction()
    add_new_mathematical_function()
    add_new_logic_symbol()
    add_new_phoneme()

    # 5. Modify existing components
    modify_computational_instruction()
    modify_mathematical_function()
    modify_logic_symbol()
    modify_phoneme()

    # 6. Remove components
    remove_computational_instruction()
    remove_mathematical_function()
    remove_logic_symbol()
    remove_phoneme()

    # 7. Serialize to JSON
    serialize_components()

    # 8. Deserialize from JSON
    deserialize_components()

    # 9. Simulated usage
    simulate_virtual_machine()
    demonstrate_logic_symbols()
    compute_mathematical_functions()
    process_phonemes()

    # 10. Additional advanced demonstrations
    nl_to_computational_instructions()
    pytorch_function_approximation()
    logical_sentiment_analysis()
    symbolic_equation_solver()
    phoneme_based_text_generation()

    # 11. (Optional) Display the log in a stylized Rich table
    process_log_file()

    logging.info("\n=== Universal Components Demo Completed ===")

    # Exit with code 1 if any ERROR-level logs exist in the log file
    with open(os.getenv("UNIVERSAL_COMPONENTS_DEMO_LOG_FILE", "universal_components_demo.log"), "r") as logcheck:
        if any('"level": "ERROR"' in line for line in logcheck):
            sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main() 
