import random
import uuid
Helper function for generating a random predicate, noun, and integer combination

    Args:
        sep: String separator for word spacing.
        integer_scale: Dictates the maximum scale range for random integer sampling (power of 10).
        max_length: Maximum allowable string length.

    Returns:
        A random string phrase comprised of a predicate, noun, and random integer.

    