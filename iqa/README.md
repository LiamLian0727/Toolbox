# Underwater Image Quality Metrics Evaluation

This repository contains Pytorch implementations for evaluating the quality of underwater images using different metrics. These metrics help assess image quality based on color, contrast, and sharpness, making them suitable for underwater photography enhancement, marine biology studies, and similar applications.

## Overview

### Key Functions

1. **UCIQE Metric**
   - `torch_uciqe(image)`: Calculates the Underwater Color Image Quality Evaluation (UCIQE) metric, which evaluates underwater images based on the standard deviation of the hue, the mean value of saturation, and luminance contrast. This function has been enhanced to support batch processing using PyTorch tensors.

2. **UIQM Metric**
   - `torch_uiqm(image)`: Computes the Underwater Image Quality Measure (UIQM) for a batch of images. The UIQM combines three components: colorfulness (UICM), sharpness (UISM), and contrast (UICONM). This metric provides a comprehensive assessment of underwater image quality.

3. **Utility Functions**
   - `sobel_torch(x)`: Performs Sobel edge detection on batch images using PyTorch.
   - `_uicm(x), _uism(x), _uiconm(x, window_size)`: Helper functions to calculate the UICM, UISM, and UICONM components used in the UIQM calculation.

### Example Usage

Here is how to calculate the UCIQE and UIQM metrics for an underwater image:

```python
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch

# Load the image and convert it to a tensor
image_path = '/path/to/your/image.png'
img = Image.open(image_path).convert('RGB')
img_tensor = to_tensor(img).cuda().unsqueeze(0)  # Convert to tensor and move to GPU if available

# Calculate the UCIQE metric
uciqe_value = torch_uciqe(img_tensor)
print(f"UCIQE value: {uciqe_value}")

# Calculate the UIQM metric
uiqm_value = torch_uiqm(img_tensor)
print(f"UIQM value: {uiqm_value}")
```

These functions are designed to handle batch operations, making them efficient for evaluating multiple images simultaneously. They leverage PyTorch for GPU acceleration, significantly reducing the time required for computation.

## Installation
To use the functions in this repository, you need the following dependencies:
- Python 3.8+
- PyTorch
- OpenCV
- Kornia
- PIL (Pillow)

Install the required libraries using:
```sh
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install opencv-python-headless kornia pillow
```

## Applications
These metrics are useful for:
- Enhancing underwater images by providing quantitative feedback.
- Comparing different image processing methods for underwater scenes.
- Marine exploration and underwater research to improve image visibility and quality.

Feel free to explore and contribute to this project by submitting pull requests or issues.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

