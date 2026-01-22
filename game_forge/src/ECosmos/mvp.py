import random
import copy
import os
import json
import signal
import time
import sys
import math
import numpy as np
import multiprocessing as mp
import asyncio
import psutil
import concurrent.futures
from datetime import datetime
import curses
from typing import List, Dict, Any, Tuple, Callable, Union, Optional
from functools import lru_cache, partial
import logging
import sympy as sp
from collections import deque
import threading

# Set up logging
logging.basicConfig(
    filename="ecosystem.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ECosmos")

# Constants - but dynamically adjustable during runtime
CONFIG = {
    "state_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "state"),
    "state_file": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "state", "mvpstate.json"
    ),
    "save_interval": 10,  # Save every N iterations
    "history_length": 1000,  # Keep last N data points
    "default_fragment_dimensions": 5,
    "interaction_radius": 0.3,  # Spatial interaction radius
    "mutation_rate": 0.01,
    "thread_count": max(1, mp.cpu_count() - 1),  # Default to N-1 cores
    "memory_limit_percent": 75,  # Max memory usage as % of system RAM
    "async_save": True,  # Use async saving
    "cache_size": 1024,  # For LRU caches
}

# Global state
RUNTIME = {
    "running": True,
    "paused": False,
    "current_iteration": 0,
    "last_save_time": time.time(),
    "active_threads": 0,
    "mem_usage": 0.0,
    "cpu_usage": 0.0,
    "mode": "normal",  # 'normal', 'turbo', 'eco'
}

# Thread pool for parallel processing
executor = None
io_executor = None
computation_executor = None

# Expression cache
_expr_cache = {}
_sympy_cache = {}
_function_cache = {}


class MathExpression:
    """Self-evolving mathematical expression system"""

    # Available operations that can evolve
    OPERATIONS = ["+", "-", "*", "/", "^", "sin", "cos", "log", "exp", "sqrt", "abs"]
    COMPLEXITY_LIMIT = 50  # Prevent expressions from becoming too complex

    def __init__(self, expression=None, variables=None):
        """Initialize with a string expression or generate a random one"""
        # Set variables FIRST before generating random expressions
        self.variables = variables or ["x", "y", "z"]

        if expression is None:
            self.expression = self._generate_random_expr(
                complexity=random.randint(1, 5)
            )
        else:
            self.expression = str(expression)

        self._compiled_fn = None
        self._sympy_expr = None
        self._complexity = self._calculate_complexity()

        # Try to compile the expression immediately
        try:
            self._compile()
        except Exception as e:
            logger.warning(f"Failed to compile expression {self.expression}: {e}")
            self.expression = "x"  # Fallback to simplest expression
            self._compile()

    def _compile(self):
        """Compile the expression into executable code for fast evaluation"""
        global _expr_cache, _sympy_cache, _function_cache

        # Use cached version if available
        if self.expression in _expr_cache:
            self._compiled_fn = _expr_cache[self.expression]
            return

        try:
            # Convert to SymPy expression for symbolic manipulation
            if self.expression in _sympy_cache:
                self._sympy_expr = _sympy_cache[self.expression]
            else:
                var_symbols = {v: sp.Symbol(v) for v in self.variables}
                expr_str = self._sanitize_expression(self.expression)
                self._sympy_expr = sp.sympify(expr_str, locals=var_symbols)
                _sympy_cache[self.expression] = self._sympy_expr

            # Compile to a lambda function
            var_dict = {v: sp.Symbol(v) for v in self.variables}
            lambda_expr = sp.lambdify(
                list(var_dict.values()), self._sympy_expr, "numpy"
            )

            # Wrap the lambda to handle errors gracefully
            def safe_eval(*args, **kwargs):
                try:
                    result = lambda_expr(*args, **kwargs)
                    if isinstance(result, (np.ndarray, list)):
                        result = np.array(result).flatten()[0]

                    # Handle NaN, inf values
                    if np.isnan(result) or np.isinf(result):
                        return 0.5  # Safe default
                    return float(result)
                except Exception:
                    return 0.5  # Return safe default on error

            self._compiled_fn = safe_eval
            _expr_cache[self.expression] = self._compiled_fn

        except Exception as e:
            logger.warning(f"Expression compilation failed: {e} for {self.expression}")
            # Fallback to a simple identity function
            self._compiled_fn = lambda x, y=0, z=0: float(x)

    def _sanitize_expression(self, expr):
        """Make the expression safe for evaluation"""
        # Replace potentially dangerous operations with safe versions
        unsafe = ["eval", "exec", "import", "__"]
        for term in unsafe:
            expr = expr.replace(term, "")
        return expr

    def evaluate(self, **kwargs):
        """Evaluate the expression with the given variable values"""
        if self._compiled_fn is None:
            self._compile()

        try:
            # Extract just the variables needed by the expression
            args = [kwargs.get(v, 0.5) for v in self.variables]
            result = self._compiled_fn(*args)

            # Ensure the result is in the [0,1] range
            result = max(0.0, min(1.0, float(result)))
            return result
        except Exception as e:
            logger.warning(f"Evaluation error: {e} for {self.expression}")
            return 0.5  # Safe default

    def mutate(self, mutation_strength=0.1):
        """Evolve the expression by mutation"""
        if random.random() < mutation_strength:
            mutation_type = random.choice(["add", "replace", "remove", "restructure"])

            if mutation_type == "add" and self._complexity < self.COMPLEXITY_LIMIT:
                # Add a new operation
                new_expr = self._add_operation(self.expression)
            elif mutation_type == "replace":
                # Replace part of the expression
                new_expr = self._replace_operation(self.expression)
            elif mutation_type == "remove" and self._complexity > 2:
                # Remove part of the expression
                new_expr = self._remove_operation(self.expression)
            else:
                # Complete restructure
                new_expr = self._generate_random_expr(complexity=random.randint(1, 5))

            # Check if we can parse the new expression
            try:
                test_expr = MathExpression(new_expr, self.variables)
                test_value = test_expr.evaluate(x=0.5, y=0.5, z=0.5)
                if not (np.isnan(test_value) or np.isinf(test_value)):
                    # Accept the mutation
                    return MathExpression(new_expr, self.variables)
            except:
                # If the mutation produces an invalid expression, don't mutate
                pass

        # Return a copy if no valid mutation
        return MathExpression(self.expression, self.variables)

    def _add_operation(self, expr):
        """Add a new operation to the expression"""
        op = random.choice(self.OPERATIONS)
        if op in ["+", "-", "*", "/"]:
            # Binary operation
            new_term = self._generate_random_expr(complexity=1)
            position = random.random()
            if position < 0.33:
                return f"({expr}) {op} ({new_term})"
            elif position < 0.66:
                return f"({new_term}) {op} ({expr})"
            else:
                # Insert into the middle if possible
                if "+" in expr or "-" in expr:
                    parts = expr.split("+" if "+" in expr else "-")
                    join_op = "+" if "+" in expr else "-"
                    insert_idx = random.randint(0, len(parts) - 1)
                    parts[insert_idx] = f"({parts[insert_idx]}) {op} ({new_term})"
                    return join_op.join(parts)
                else:
                    return f"({expr}) {op} ({new_term})"
        else:
            # Unary operation
            return f"{op}({expr})"

    def _replace_operation(self, expr):
        """Replace a part of the expression"""
        # Simple replacement strategy - in a real system, this would be more sophisticated
        # using proper parsing of the expression tree
        op = random.choice(self.OPERATIONS)
        var = random.choice(self.variables)

        # Find a suitable replacement target
        targets = []
        for i, char in enumerate(expr):
            if (
                char in "+-*/^"
                or expr[i : i + 3] in ["sin", "cos", "log", "exp", "abs"]
                or char in self.variables
            ):
                targets.append(i)

        if not targets:
            return expr

        pos = random.choice(targets)
        if expr[pos] in "+-*/^":
            # Replace binary operator
            return expr[:pos] + op + expr[pos + 1 :] if op in "+-*/^" else expr
        elif expr[pos] in self.variables:
            # Replace variable
            return expr[:pos] + random.choice(self.variables) + expr[pos + 1 :]
        else:
            # Replace function name (care needed here in a real system)
            for func in ["sin", "cos", "log", "exp", "abs", "sqrt"]:
                if expr[pos : pos + len(func)] == func:
                    new_func = random.choice(
                        ["sin", "cos", "log", "exp", "abs", "sqrt"]
                    )
                    return expr[:pos] + new_func + expr[pos + len(func) :]

        return expr

    def _remove_operation(self, expr):
        """Simplify by removing a part of the expression"""
        # This is a simple approach - a real system would use proper expression parsing
        if "+" in expr:
            parts = expr.split("+")
            if len(parts) > 1:
                remove_idx = random.randint(0, len(parts) - 1)
                return "+".join(parts[:remove_idx] + parts[remove_idx + 1 :])
        elif "*" in expr:
            parts = expr.split("*")
            if len(parts) > 1:
                remove_idx = random.randint(0, len(parts) - 1)
                return "*".join(parts[:remove_idx] + parts[remove_idx + 1 :])

        # If we can't easily simplify, return a variable
        return random.choice(self.variables)

    def _generate_random_expr(self, complexity=3):
        """Generate a random mathematical expression"""
        if complexity <= 1:
            # Base case: just a variable or constant
            if random.random() < 0.8:
                return random.choice(self.variables)
            else:
                return str(round(random.random(), 2))

        # Recursive case: build a more complex expression
        op = random.choice(self.OPERATIONS)
        if op in ["+", "-", "*", "/"]:
            # Binary operation
            left = self._generate_random_expr(complexity=complexity - 1)
            right = self._generate_random_expr(complexity=complexity - 1)
            return f"({left}){op}({right})"
        else:
            # Unary operation
            inner = self._generate_random_expr(complexity=complexity - 1)
            return f"{op}({inner})"

    def _calculate_complexity(self):
        """Calculate the complexity of the expression"""
        # Simple heuristic based on length and number of operations
        complexity = len(self.expression)
        for op in self.OPERATIONS:
            complexity += self.expression.count(op) * 2
        return complexity

    def __str__(self):
        return self.expression

    def to_dict(self):
        return {"expression": self.expression, "variables": self.variables}

    @classmethod
    def from_dict(cls, data):
        return cls(data["expression"], data["variables"])


class SpatialVector:
    """N-dimensional vector for spatial positioning of fragments"""

    def __init__(self, dimensions=3, values=None):
        self.dimensions = dimensions
        if values is None:
            self.values = np.random.random(dimensions)
        else:
            self.values = np.array(values, dtype=float)
            if len(self.values) != dimensions:
                self.values = np.resize(self.values, dimensions)

    def distance(self, other):
        """Calculate Euclidean distance to another vector"""
        return np.linalg.norm(self.values - other.values)

    def move_towards(self, other, step=0.1):
        """Move this vector towards another vector"""
        direction = other.values - self.values
        length = np.linalg.norm(direction)
        if length > 0:
            normalized = direction / length
            self.values += normalized * step

    def mutate(self, strength=0.1):
        """Mutate the vector"""
        mutation = np.random.normal(0, strength, self.dimensions)
        new_values = self.values + mutation
        # Optional: Apply constraints like wrapping around a toroidal space
        new_values = new_values % 1.0
        return SpatialVector(self.dimensions, new_values)

    def to_dict(self):
        return {"dimensions": self.dimensions, "values": self.values.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["dimensions"], data["values"])


class EvolutionaryRuleSet:
    """A set of rules that can evolve over time"""

    def __init__(self, rule_count=5, rules=None, meta_rules=None):
        self.rule_count = rule_count

        if rules is None:
            # Generate random rules
            self.rules = [self._create_rule() for _ in range(rule_count)]
        else:
            self.rules = rules.copy()

        # Meta-rules govern how rules themselves evolve
        if meta_rules is None:
            self.meta_rules = {
                "mutation_rate": MathExpression("0.05"),
                "crossover_rate": MathExpression("0.02"),
                "complexity_preference": MathExpression("sin(x*3.14)"),
                "adaptation_rate": MathExpression("0.01"),
            }
        else:
            self.meta_rules = meta_rules

    def _create_rule(self):
        """Create a new rule as a mathematical expression"""
        return MathExpression()

    def apply_rules(self, inputs, context=None):
        """Apply the ruleset to generate outputs"""
        results = []
        context = context or {}

        for rule in self.rules:
            # Prepare the environment for rule evaluation
            env = inputs.copy()
            env.update(context)

            try:
                result = rule.evaluate(**env)
                results.append(result)
            except Exception as e:
                logger.warning(f"Rule application error: {e}")
                results.append(0.5)  # Safe default

        return np.array(results)

    def evolve(self, fitness=0.5, external_influence=0.0):
        """Evolve the ruleset based on fitness and external influence"""
        new_rules = []

        # Get mutation parameters from meta-rules
        try:
            mutation_rate = self.meta_rules["mutation_rate"].evaluate(
                x=fitness, y=external_influence, z=len(self.rules) / 10
            )
            complexity_pref = self.meta_rules["complexity_preference"].evaluate(
                x=fitness, y=external_influence, z=0.5
            )
        except Exception:
            mutation_rate = 0.05
            complexity_pref = 0.5

        # Evolve each rule
        for rule in self.rules:
            if random.random() < mutation_rate:
                # Mutate this rule
                new_rule = rule.mutate(mutation_strength=mutation_rate * 2)
                new_rules.append(new_rule)
            else:
                new_rules.append(rule)

        # Occasionally evolve meta-rules themselves
        if random.random() < 0.01:
            meta_rule_key = random.choice(list(self.meta_rules.keys()))
            self.meta_rules[meta_rule_key] = self.meta_rules[meta_rule_key].mutate(0.05)

        # Occasionally add or remove rules
        if random.random() < 0.02:
            if random.random() < complexity_pref and len(new_rules) < 20:
                # Add a rule
                new_rules.append(self._create_rule())
                self.rule_count += 1
            elif len(new_rules) > 1:
                # Remove a rule
                del new_rules[random.randint(0, len(new_rules) - 1)]
                self.rule_count -= 1

        return EvolutionaryRuleSet(self.rule_count, new_rules, self.meta_rules)

    def to_dict(self):
        return {
            "rule_count": self.rule_count,
            "rules": [rule.to_dict() for rule in self.rules],
            "meta_rules": {k: v.to_dict() for k, v in self.meta_rules.items()},
        }

    @classmethod
    def from_dict(cls, data):
        rules = [MathExpression.from_dict(r) for r in data["rules"]]
        meta_rules = {
            k: MathExpression.from_dict(v) for k, v in data["meta_rules"].items()
        }
        return cls(data["rule_count"], rules, meta_rules)


class Fragment:
    """
    A fundamental unit in our universe - now with evolved mathematics and behaviors.
    All properties and behaviors emerge from interactions between components.
    """

    def __init__(self, state=None, position=None, ruleset=None):
        # Initialize with provided or random state
        if state is None:
            self.state = {
                "values": np.random.random(
                    CONFIG["default_fragment_dimensions"]
                ).tolist(),
                "properties": {},  # Emergent properties
                "age": 0,  # Track fragment lifetime
                "lineage": random.randint(
                    0, 1000000
                ),  # For tracking evolutionary lines
                "memory": {},  # Fragments can retain information
            }
        else:
            self.state = state

        # Spatial position in the universe
        self.position = position or SpatialVector()

        # Rules that govern behavior and evolution
        self.ruleset = ruleset or EvolutionaryRuleSet()

        # Internal properties cache for performance
        self._property_cache = {}
        self._last_update = 0

        # Initialize properties
        self._update_properties()

    @lru_cache(maxsize=1024)
    def _calculate_property(self, prop_name, values):
        """Calculate an emergent property from values"""
        # Convert values list to hashable tuple for caching
        values_tuple = tuple(values)

        # Different properties emerge from different combinations of values
        if prop_name == "stability":
            # Stability emerges from the balance of values
            return sum(abs(v - 0.5) for v in values) / len(values)
        elif prop_name == "complexity":
            # Complexity from the variance and number of dimensions
            return (np.var(values) * math.log(1 + len(values))) % 1.0
        elif prop_name == "energy":
            # Energy from the sum of values
            return sum(values) / len(values)
        elif prop_name == "coherence":
            # Coherence from how ordered the values are
            sorted_vals = sorted(values)
            diffs = [
                abs(sorted_vals[i] - sorted_vals[i - 1])
                for i in range(1, len(sorted_vals))
            ]
            return 1.0 - (sum(diffs) / len(diffs) if diffs else 0)
        elif prop_name == "adaptability":
            # Adaptability from distribution of values
            return 1.0 - abs(np.mean(values) - 0.5)
        else:
            # Default property calculation
            return np.mean(values)

    def _update_properties(self):
        """Update emergent properties based on current state"""
        values = self.state["values"]

        # Only recalculate properties every so often for performance
        if self._last_update == RUNTIME["current_iteration"]:
            return

        self._last_update = RUNTIME["current_iteration"]

        # Calculate basic emergent properties
        self.state["properties"]["stability"] = self._calculate_property(
            "stability", tuple(values)
        )
        self.state["properties"]["complexity"] = self._calculate_property(
            "complexity", tuple(values)
        )
        self.state["properties"]["energy"] = self._calculate_property(
            "energy", tuple(values)
        )
        self.state["properties"]["coherence"] = self._calculate_property(
            "coherence", tuple(values)
        )
        self.state["properties"]["adaptability"] = self._calculate_property(
            "adaptability", tuple(values)
        )

        # Meta-property: fitness combines multiple properties
        props = self.state["properties"]
        self.state["properties"]["fitness"] = (
            props["stability"] * 0.3
            + props["complexity"] * 0.2
            + props["energy"] * 0.15
            + props["coherence"] * 0.2
            + props["adaptability"] * 0.15
        )

    def interact(self, others):
        """
        Enhanced interaction function between fragments.
        Now uses evolved rules and spatial relationships.
        """
        # Update age
        self.state["age"] = self.state.get("age", 0) + 1

        # Skip interactions if no others or no rules
        if not others or not self.ruleset.rules:
            mutation_result = self._mutate(0.1)
            mutation_result._update_properties()
            return mutation_result

        # Make a deep copy for modification
        new_state = copy.deepcopy(self.state)
        new_position = copy.deepcopy(self.position)

        # Prepare inputs for rule application
        inputs = {
            "x": np.mean(self.state["values"]),
            "y": self.state["age"] / 1000.0,
            "z": len(others) / 10.0,
        }

        # Add information about nearest neighbors
        nearest = sorted(others, key=lambda f: self.position.distance(f.position))[:3]
        if nearest:
            inputs["nearest_dist"] = self.position.distance(nearest[0].position)
            inputs["nearest_value"] = np.mean(nearest[0].state["values"])

        # Calculate external influence
        external_values = []
        for other in others:
            # Weight influence by distance
            dist = self.position.distance(other.position)
            if dist < CONFIG["interaction_radius"]:
                weight = 1.0 - (dist / CONFIG["interaction_radius"])
                if len(other.state["values"]) == len(self.state["values"]):
                    # Direct influence if dimensions match
                    weighted_values = np.array(other.state["values"]) * weight
                    external_values.append(weighted_values)
                else:
                    # Just add mean value otherwise
                    external_values.append(
                        np.array([np.mean(other.state["values"])]) * weight
                    )

        # Apply rules to generate new values
        if external_values:
            # Calculate net external influence
            ext_influence = np.mean(external_values, axis=0)

            # Ensure dimensions match
            if len(ext_influence) != len(self.state["values"]):
                ext_influence = np.resize(ext_influence, len(self.state["values"]))

            # Add external influence to inputs
            inputs["ext"] = float(np.mean(ext_influence))

            # Apply ruleset to generate new values
            rule_outputs = self.ruleset.apply_rules(inputs)

            # Use rule outputs to modify state values
            for i, val in enumerate(new_state["values"]):
                if i < len(rule_outputs):
                    # Complex update rule
                    new_val = (
                        val * (1.0 - rule_outputs[i])
                        + ext_influence[i % len(ext_influence)] * rule_outputs[i]
                    )
                    new_state["values"][i] = float(new_val)

            # Spatial movement based on attraction/repulsion
            # FIX: Check array length instead of truthiness
            move_factor = rule_outputs[0] if len(rule_outputs) > 0 else 0.5
            if move_factor > 0.7:  # Attraction
                if nearest:
                    new_position.move_towards(nearest[0].position, step=0.05)
            elif move_factor < 0.3:  # Repulsion
                if nearest:
                    # Move away
                    temp = copy.deepcopy(new_position)
                    temp.move_towards(nearest[0].position, step=0.05)
                    diff = temp.values - new_position.values
                    new_position.values -= diff

        # Ensure values stay in valid range
        new_state["values"] = [max(0.0, min(1.0, v)) for v in new_state["values"]]

        # Memory updates - fragments can "remember" past interactions
        if not "memory" in new_state:
            new_state["memory"] = {}

        # Store memory of most significant interaction
        if nearest:
            most_influential = max(nearest, key=lambda f: f.get_property("fitness", 0))
            memory_key = f"encountered_{most_influential.state.get('lineage', 0)}"
            new_state["memory"][memory_key] = new_state["memory"].get(memory_key, 0) + 1

        # Evolve ruleset based on fitness and external influence
        fitness = self.get_property("fitness", 0.5)
        ext_avg = inputs.get("ext", 0)
        new_ruleset = self.ruleset.evolve(fitness, ext_avg)

        # Create the next generation fragment
        result = Fragment(new_state, new_position, new_ruleset)

        # Apply random mutations
        result = result._mutate(CONFIG["mutation_rate"])

        # Update properties on the new fragment
        result._update_properties()

        return result

    def _mutate(self, rate):
        """Apply random mutations to the fragment"""
        if random.random() > rate:
            return self  # No mutation

        new_state = copy.deepcopy(self.state)
        new_position = self.position.mutate(rate)

        # Value mutations
        for i in range(len(new_state["values"])):
            if random.random() < rate:
                # Apply mutation
                mutation = random.gauss(0, 0.1)
                new_state["values"][i] = max(
                    0, min(1, new_state["values"][i] + mutation)
                )

        # Structural mutations (add/remove dimensions)
        if random.random() < rate * 0.2:
            if random.random() < 0.5 and len(new_state["values"]) > 1:
                # Remove a dimension
                idx = random.randint(0, len(new_state["values"]) - 1)
                new_state["values"].pop(idx)
            else:
                # Add a dimension
                new_state["values"].append(random.random())

        # Lineage tracking - mutations create a new branch
        if random.random() < rate:
            new_state["lineage"] = random.randint(0, 1000000)

        return Fragment(new_state, new_position, self.ruleset)

    def get_property(self, name, default=0.0):
        """Get a property with fallback default"""
        self._update_properties()  # Ensure properties are current
        return self.state.get("properties", {}).get(name, default)

    def stability(self):
        """Legacy method for compatibility"""
        return self.get_property("fitness", 0.5)

    def to_dict(self):
        """Convert to serializable dict for state saving"""
        return {
            "state": copy.deepcopy(self.state),
            "position": self.position.to_dict(),
            "ruleset": self.ruleset.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        """Create Fragment from saved state dict"""
        position = SpatialVector.from_dict(data["position"])
        ruleset = EvolutionaryRuleSet.from_dict(data["ruleset"])
        return cls(data["state"], position, ruleset)

    def __str__(self):
        """String representation"""
        props = self.state.get("properties", {})
        props_str = ", ".join(f"{k}:{v:.2f}" for k, v in props.items())
        values_str = "[" + ", ".join(f"{v:.2f}" for v in self.state["values"]) + "]"
        return f"Fragment(values={values_str}, properties={props_str})"


def save_state(universe, iteration, stats):
    """Save the current universe state to a file"""
    if not os.path.exists(CONFIG["state_dir"]):
        os.makedirs(CONFIG["state_dir"])

    state_data = {
        "iteration": iteration,
        "universe": [f.to_dict() for f in universe],
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
    }

    # Use a temporary file to avoid corruption on interrupt
    temp_file = CONFIG["state_file"] + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(state_data, f)

    # Atomic rename for safety
    os.replace(temp_file, CONFIG["state_file"])


def load_state():
    """Load universe state from file if it exists"""
    if not os.path.exists(CONFIG["state_file"]):
        return None, 0, {}

    try:
        with open(CONFIG["state_file"], "r") as f:
            state_data = json.load(f)

        universe = [Fragment.from_dict(f_dict) for f_dict in state_data["universe"]]
        iteration = state_data.get("iteration", 0)
        stats = state_data.get("stats", {})

        return universe, iteration, stats
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading state: {e}")
        return None, 0, {}


def handle_signal(signum, frame):
    """Handle interruption signals gracefully"""
    global running
    running = False


def calculate_stats(universe, old_stats=None):
    """Calculate statistics about the universe with enhanced metrics"""
    if not universe:
        return old_stats or {}

    if old_stats is None:
        old_stats = {}

    # Ensure required keys exist in stats dictionary
    for key in [
        "stability_history",
        "size_history",
        "complexity_history",
        "rule_complexity_history",
        "expression_examples",
        "property_distributions",
        "most_fit_lineage",
        "dimension_history",
        "age_distribution",
    ]:
        if key not in old_stats:
            old_stats[key] = []

    if "mutation_rate" not in old_stats:
        old_stats["mutation_rate"] = CONFIG["mutation_rate"]

    # Calculate current stats
    stabilities = [f.stability() for f in universe]
    complexities = [len(f.state["values"]) for f in universe]
    complexity_avg = sum(complexities) / len(universe) if universe else 0

    # Rule complexity - how complex are the expressions?
    rule_complexities = []
    expressions = []
    for f in universe[:10]:  # Sample from top fragments
        for rule in f.ruleset.rules[:3]:  # Sample a few rules
            rule_complexities.append(rule._complexity)
            expressions.append(str(rule))

    rule_complexity_avg = (
        sum(rule_complexities) / len(rule_complexities) if rule_complexities else 0
    )

    # Property distributions
    property_keys = ["stability", "complexity", "energy", "coherence", "adaptability"]
    property_avgs = {}
    for key in property_keys:
        values = [f.get_property(key, 0) for f in universe]
        property_avgs[key] = sum(values) / len(values) if values else 0

    # Track most successful lineage
    if universe:
        most_fit = max(universe, key=lambda f: f.stability())
        lineage = most_fit.state.get("lineage", 0)
        old_stats["most_fit_lineage"] = lineage

    # Age distribution
    ages = [f.state.get("age", 0) for f in universe]
    age_avg = sum(ages) / len(ages) if ages else 0

    # Update histories (keep last N points based on CONFIG)
    history_length = CONFIG.get("history_length", 100)
    old_stats["stability_history"] = (
        old_stats["stability_history"]
        + [sum(stabilities) / len(stabilities) if stabilities else 0]
    )[-history_length:]
    old_stats["size_history"] = (old_stats["size_history"] + [len(universe)])[
        -history_length:
    ]
    old_stats["complexity_history"] = (
        old_stats["complexity_history"] + [complexity_avg]
    )[-history_length:]
    old_stats["rule_complexity_history"] = (
        old_stats["rule_complexity_history"] + [rule_complexity_avg]
    )[-history_length:]
    old_stats["dimension_history"] = (
        old_stats["dimension_history"] + [sum(complexities)]
    )[-history_length:]

    # Save interesting expressions for display
    if expressions:
        old_stats["expression_examples"] = expressions[
            :5
        ]  # Keep a few interesting expressions

    # Store current property distributions
    old_stats["property_distributions"] = property_avgs
    old_stats["age_distribution"] = age_avg

    return old_stats


def init_curses():
    """Initialize curses for the terminal UI"""
    stdscr = curses.initscr()
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_RED, -1)
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    curses.curs_set(0)  # Hide cursor
    return stdscr


def end_curses(stdscr):
    """Cleanup curses on exit"""
    stdscr.keypad(False)
    curses.nocbreak()
    curses.echo()
    curses.endwin()


def render_ui(stdscr, universe, iteration, stats):
    """Render enhanced terminal UI with more detailed metrics"""
    height, width = stdscr.getmaxyx()
    stdscr.clear()

    # Safety function to prevent writing outside screen bounds
    def safe_addstr(y, x, text, attr=0):
        """Safely add a string, checking bounds"""
        if y >= height or x >= width:
            return

        # Truncate text if it would exceed screen width
        remaining_width = width - x
        if len(text) > remaining_width:
            text = text[:remaining_width]

        try:
            stdscr.addstr(y, x, text, attr)
        except curses.error:
            # Still handle any other curses errors gracefully
            pass

    # Title and iteration
    title = "Universal Fragment Simulator"
    safe_addstr(
        0, (width - len(title)) // 2, title, curses.A_BOLD | curses.color_pair(5)
    )
    safe_addstr(1, 0, f"Iteration: {iteration}", curses.A_BOLD)
    safe_addstr(1, width - 20, f"Fragments: {len(universe)}", curses.color_pair(2))

    # Top fragments
    safe_addstr(
        3, 0, "Top Stable Fragments:", curses.A_UNDERLINE | curses.color_pair(1)
    )
    for i, frag in enumerate(
        sorted(universe, key=lambda f: -f.stability())[:4]
    ):  # Show fewer fragments
        if 4 + i * 4 + 3 >= height:  # Check if we'll exceed screen height
            break

        stability_color = 3 if frag.stability() > 0.5 else 4
        v_str = "[" + ", ".join(f"{v:.2f}" for v in frag.state["values"]) + "]"

        # Truncate to fit screen
        if len(v_str) > width - 12:
            v_str = v_str[: width - 15] + "...]"

        # Show top rules from the fragment
        rules = frag.ruleset.rules[:2]  # Just show 2 rules to save space
        r_str = "[" + ", ".join(str(rule)[:15] for rule in rules) + "...]"

        frag_str = (
            f"{i+1}. Fitness: {frag.stability():.3f}  Age: {frag.state.get('age', 0)}"
        )
        values_str = f"Values: {v_str}"
        rules_str = f"Rules: {r_str}"

        # Add lineage info
        lineage_str = f"Lineage: {frag.state.get('lineage', 0)}"

        safe_addstr(4 + i * 4, 2, frag_str, curses.color_pair(stability_color))
        safe_addstr(5 + i * 4, 4, values_str)
        safe_addstr(6 + i * 4, 4, rules_str)
        safe_addstr(7 + i * 4, 4, lineage_str, curses.color_pair(5))

    # Statistics section - ensure we don't go beyond screen bounds
    stats_y = min(height - 10, 4 + 4 * min(len(universe), 4) + 1)

    if stats_y < height - 1:  # Make sure we have at least one line left
        safe_addstr(
            stats_y, 0, "System Statistics:", curses.A_UNDERLINE | curses.color_pair(1)
        )

        # Property distributions as bar charts
        if stats and "property_distributions" in stats:
            props = stats["property_distributions"]
            for i, (prop, val) in enumerate(props.items()):
                if stats_y + i + 1 >= height - 2:  # Reserve space for help text
                    break

                # Show property name and bar
                bar_len = min(
                    int(val * 20), width - 25
                )  # Ensure bar doesn't exceed screen
                bar_color = 1 if val > 0.5 else 4
                safe_addstr(stats_y + i + 1, 2, f"{prop.capitalize()}: {val:.2f}")
                safe_addstr(
                    stats_y + i + 1, 20, "█" * bar_len, curses.color_pair(bar_color)
                )

    # Help text - always at bottom
    safe_addstr(
        height - 1, 0, "Press 'q' to save and quit, 's' to save state", curses.A_DIM
    )

    stdscr.refresh()


def run_universe(stdscr, num_fragments=32):
    """
    Run the universe simulation indefinitely with visualization.
    This is the main simulation loop.
    """
    global running

    # Try to load saved state
    universe, iteration, stats = load_state()
    if universe is None:
        # Initialize a "soup" of fragments
        universe = [Fragment() for _ in range(num_fragments)]
        iteration = 0
        stats = {}

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Ensure non-blocking getch
    stdscr.nodelay(True)

    # Initialize stats if needed
    stats = calculate_stats(universe, stats)

    # Main simulation loop - run indefinitely until interrupted
    last_save_time = time.time()

    running = True

    # Initialize with proper runtime tracking values
    RUNTIME["current_iteration"] = iteration

    while running:
        # Check for user input
        try:
            key = stdscr.getch()
            if key == ord("q"):
                running = False
            elif key == ord("s"):
                save_state(universe, iteration, stats)
                stdscr.addstr(
                    0, 0, "State saved!", curses.A_BOLD | curses.color_pair(3)
                )
                stdscr.refresh()
                time.sleep(1)
        except:
            pass

        # Let fragments interact with each other
        new_universe = []
        for frag in universe:
            # Select a random subset of other fragments
            others = (
                random.sample(
                    [f for f in universe if f != frag], k=min(len(universe) - 1, 3)
                )
                if len(universe) > 1
                else []
            )

            # Fragment interacts with others
            new_universe.append(frag.interact(others))

        # Resource constraint: only some fragments survive
        # but the selection criteria emerges from their state
        universe = sorted(new_universe, key=lambda f: f.stability())
        keep_size = max(1, len(universe) // 2)
        universe = universe[:keep_size]

        # Occasionally introduce new random fragments (influx of energy/matter)
        if random.random() < 0.05:
            universe.append(Fragment())

        # Update statistics
        stats = calculate_stats(universe, stats)

        # Render UI
        render_ui(stdscr, universe, iteration, stats)

        # Auto-save state periodically
        current_time = time.time()
        if current_time - last_save_time > CONFIG["save_interval"]:
            save_state(universe, iteration, stats)
            last_save_time = current_time

        # Update current iteration in global state for metrics calculation
        RUNTIME["current_iteration"] = iteration

        iteration += 1
        time.sleep(0.1)  # Small delay to prevent CPU hogging

    # Save state on exit
    save_state(universe, iteration, stats)
    return universe


def main():
    """Main entry point with curses wrapper for clean terminal handling"""
    try:
        # Initialize math module which I'm using in the Fragment class
        import math

        # Initialize curses
        stdscr = init_curses()

        try:
            # Run simulation
            universe = run_universe(stdscr)
        finally:
            # Ensure curses cleanup happens
            end_curses(stdscr)

        # Print final summary to normal terminal
        print("\nSimulation ended. Final universe:")
        for i, f in enumerate(sorted(universe, key=lambda f: -f.stability())[:5]):
            print(f"{i}: {f}")

    except Exception as e:
        # Make sure terminal is usable even if there's an error
        try:
            curses.endwin()
        except:
            pass
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
