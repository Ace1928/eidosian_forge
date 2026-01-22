import base64
import mimetypes
def image_to_data_url(image_path: str) -> str:
    encoding = encode_image(image_path)
    mime_type = mimetypes.guess_type(image_path)[0]
    return f'data:{mime_type};base64,{encoding}'