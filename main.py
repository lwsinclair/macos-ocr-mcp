import io
import os
import platform

from mcp.server.fastmcp import FastMCP
from ocrmac import ocrmac as ocrmac_module
from PIL import Image

mcp = FastMCP()


@mcp.tool()
async def ocr_image(file_path: str) -> dict:
    """
    Conduct macOS built-in OCR and return the text
    """
    if platform.system() != 'Darwin':
        return {'error': 'OCR functionality is only available on macOS.'}

    try:
        with open(file_path, 'rb') as f:
            contents = f.read()
        pil_image = Image.open(io.BytesIO(contents))
        raw_annotations = ocrmac_module.OCR(pil_image).recognize()

        # Process annotations to include text, confidence, and bounding box
        processed_annotations = [
            {'text': ann[0], 'confidence': ann[1], 'bounding_box': ann[2]}
            for ann in raw_annotations
        ]

        filename = os.path.basename(file_path)
        return {'filename': filename, 'annotations': processed_annotations}
    except FileNotFoundError:
        return {'error': f'File not found: {file_path}'}
    except Exception as e:
        return {'error': f'Error processing image: {str(e)}'}


if __name__ == '__main__':
    mcp.run()
