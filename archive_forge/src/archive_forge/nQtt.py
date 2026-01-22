from EVIE.standard_decorator import StandardDecorator


@StandardDecorator()
def get_tile_color(value: int) -> tuple:
    """
    Generates a color for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        tuple: The color (R, G, B) for the tile.
    """
    # Example logic for generating a color based on the tile's value
    # This is a placeholder and should be replaced with a dynamic gradient generator
    base_color = (238, 228, 218)
    factor = log(value, 2) * 20  # Simplified example for color variation
    return tuple(min(max(0, x + int(factor)), 255) for x in base_color)
