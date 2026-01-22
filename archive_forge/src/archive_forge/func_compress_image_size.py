from typing import TYPE_CHECKING, Optional, Tuple, Union
def compress_image_size(image: 'PIL.Image.Image', max_size: Optional[int]=COMPRESSED_IMAGE_SIZE) -> 'PIL.Image.Image':
    """
    Scale the image to fit within a square with length `max_size` while maintaining
    the aspect ratio.
    """
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (new_width / width))
    else:
        new_height = max_size
        new_width = int(width * (new_height / height))
    return image.resize((new_width, new_height))