import os
import cv2
import math
import time

import numpy as np
import kornia.color as color
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor


def uciqe(image):
    image = cv2.imread(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # RGB转为HSV
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180  # Standard deviation of colorimetry
    mu = np.mean(S) / 255  # Average of saturation
    n, m = np.shape(V)  # Get the luminance contrast value
    number = math.floor(n * m / 100)
    v = V.flatten() / 255
    v.sort()
    bottom = np.sum(v[:number]) / number
    v = -v
    v.sort()
    v = -v
    top = np.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    print(f"delta: {delta},  conl: {conl}, mu: {mu}")
    return uciqe


# improve the following code to support batch operation:

def torch_uciqe(image):
    """
        Calculate the Underwater Color Image Quality Evaluation (UCIQE) metric for a given image.

        Parameters:
        image (torch.Tensor): A 4D tensor representing the batch of images in RGB format with shape (B, 3, H, W).
                              The image values should be in the range [0, 1].

        Returns:
        float: The UCIQE value with shape (B), which is a measure of the quality of underwater images.

        The UCIQE metric combines three components:
        - delta: The standard deviation of the hue channel (H) divided by 2 * pi.
        - mu:    The mean value of the saturation channel (S).
        - conl:  The contrast of the luminance channel (V), calculated as the difference between the
                 top 1% and bottom 1% values.

        Example usage:
        ```python
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        import torch

        # Load the image and convert to tensor
        image_path = '/path/to/your/image.png'
        img = Image.open(image_path).convert('RGB')
        img_tensor = to_tensor(img).cuda().unsqueeze(0)  # Convert to tensor and move to GPU if available

        # Calculate the UCIQE metric
        uciqe_value = torch_uciqe(img_tensor)
        print(f"UCIQE value: {uciqe_value}")
        ```
    """

    hsv = color.rgb_to_hsv(image)
    H, S, V = torch.chunk(hsv, 3, dim=1)
    # print(H.shape, S.shape, V.shape)

    # Standard deviation of colorimetry
    delta = torch.std(H, dim=(1, 2, 3)) / (2 * math.pi)

    # Average of saturation
    mu = torch.mean(S, dim=(1, 2, 3))

    # Get the luminance contrast value
    batch_size, _, n, m = V.shape
    number = math.floor(n * m / 100)
    v = V.view(batch_size, -1)  # Flatten each image in the batch
    v, _ = v.sort(dim=1)
    bottom = torch.sum(v[:, :number], dim=1) / number
    top = torch.sum(v[:, -number:], dim=1) / number
    conl = top - bottom
    # Calculate UCIQE
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    print(f"delta: {delta},  conl: {conl}, mu: {mu}")
    return uciqe



def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
       Calculates the asymmetric alpha-trimmed mean for batched input
    """
    # sort pixels by intensity - for clipping
    x, _ = torch.sort(x, dim=1)
    # get number of pixels
    K = x.shape[1]
    # calculate T alpha L and T alpha R
    alpha_L = torch.tensor(alpha_L, dtype=torch.float32)
    alpha_R = torch.tensor(alpha_R, dtype=torch.float32)
    T_a_L = torch.ceil(alpha_L * K).to(torch.int32)
    T_a_R = torch.floor(alpha_R * K).to(torch.int32)
    # calculate mu_alpha weight
    weight = (1 / (K - T_a_L - T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s = T_a_L
    e = K - T_a_R
    val = torch.sum(x[:, s:e], dim=1)
    val = weight * val
    return val


def s_a(x, mu):
    """
      Calculates the variance for batched input
    """
    val = torch.sum(torch.pow(x - mu.unsqueeze(1), 2), dim=1) / x.shape[1]
    return val


def _uicm(x):
    batch_size = x.shape[0]
    R = x[:, 0, :, :].reshape(batch_size, -1)
    G = x[:, 1, :, :].reshape(batch_size, -1)
    B = x[:, 2, :, :].reshape(batch_size, -1)
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = torch.sqrt((torch.pow(mu_a_RG, 2) + torch.pow(mu_a_YB, 2)))
    r = torch.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)


sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)


def sobel_torch(x):
    """
    Sobel edge detection for batch input
    """
    batch_size = x.shape[0]
    dx = F.conv2d(x[:, None], sobel_kernel_x.to(x.device), padding=1)
    dy = F.conv2d(x[:, None], sobel_kernel_y.to(x.device), padding=1)
    mag = torch.hypot(dx, dy)
    mag *= 255.0 / torch.amax(mag, dim=(1, 2, 3), keepdim=True)
    return mag.squeeze(1)

def eme(x, window_size):
    """
    Enhancement measure estimation for batch input
    """
    batch_size = x.shape[0]
    k1 = x.shape[2] // window_size
    k2 = x.shape[1] // window_size
    x = x[:, :k2 * window_size, :k1 * window_size]

    # Reshape x into a tensor with shape (batch_size, k2, window_size, k1, window_size)
    x = x.view(batch_size, k2, window_size, k1, window_size)

    # Transpose and reshape the tensor into shape (batch_size, k2*k1, window_size*window_size)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, -1, window_size * window_size)

    # Compute the max and min values for each block
    max_vals, _ = torch.max(x, dim=2)
    min_vals, _ = torch.min(x, dim=2)

    # Bound checks, can't do log(0)
    non_zero_mask = (min_vals != 0) & (max_vals != 0)

    # Compute the log ratios
    log_ratios = torch.zeros_like(max_vals)
    log_ratios[non_zero_mask] = torch.log(max_vals[non_zero_mask] / min_vals[non_zero_mask])

    # Compute the sum of the log ratios
    val = log_ratios.sum(dim=1)

    # Compute the weight
    w = 2. / (k1 * k2)

    return w * val

def _uism(x):
    """
    Underwater Image Sharpness Measure for batch input
    """
    batch_size = x.shape[0]
    channel_R = x[:, 0, :, :]
    channel_G = x[:, 1, :, :]
    channel_B = x[:, 2, :, :]

    # Apply Sobel edge detector to each RGB component
    Rs = sobel_torch(channel_R)
    Gs = sobel_torch(channel_G)
    Bs = sobel_torch(channel_B)

    # Multiply the edges detected for each channel by the channel itself
    R_edge_map = torch.multiply(Rs, channel_R)
    G_edge_map = torch.multiply(Gs, channel_G)
    B_edge_map = torch.multiply(Bs, channel_B)

    # Get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)

    # Coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144

    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)


def _uiconm(x, window_size):
    # Ensure image is divisible by window_size - doesn't matter if we cut out some pixels
    k1 = x.shape[3] // window_size
    k2 = x.shape[2] // window_size
    x = x[:, :, :k2 * window_size, :k1 * window_size]

    # Weight
    w = -1. / (k1 * k2)

    # Entropy scale - higher helps with randomness
    alpha = 1

    # Create blocks
    # 3, 108, 192, 10, 10
    x = x.unfold(2, window_size, window_size).unfold(3, window_size, window_size)

    # Compute min and max values for each block
    min_ = x.min(dim=-1).values.min(dim=-1).values.min(dim=1).values  # shape: (batch_size, k2, k1)
    max_ = x.max(dim=-1).values.max(dim=-1).values.max(dim=1).values  # shape: (batch_size, k2, k1)

    # Calculate top and bot
    top = max_ - min_
    bot = max_ + min_
    print(min_.shape, max_.shape)

    # Calculate the value for each block
    val = alpha * torch.pow((top / bot), alpha) * torch.log(top / bot)

    # Handle NaN and zero values
    val = torch.where(torch.isnan(val) | (bot == 0.0) | (top == 0.0), torch.zeros_like(val), val)

    # Sum up the values and apply the weight
    val = w * val.sum(dim=[1, 2])

    return val


# Replace all instances of torch with torch and ndimage, sobel with sobel_torch in the original code
# Also, convert the image to tensor at the beginning and back to NumPy array at the end of the function
def torch_uiqm(x):
    """
        Calculate the Underwater Image Quality Measure (UIQM) for a given image.

        Parameters:
        x (torch.Tensor): A 4D tensor representing the images in RGB format with shape (B, 3, H, W).
                          The image values should be in the range [0, 1].

        Returns:
        float: The UIQM value, which is a measure of the overall quality of underwater images.

        The UIQM metric combines three components:
        - UICM (Underwater Image Colorfulness Measure)
        - UISM (Underwater Image Sharpness Measure)
        - UICONM (Underwater Image Contrast Measure)

        The metric is calculated using the formula:
        UIQM = c1 * UICM + c2 * UISM + c3 * UICONM,
        where c1, c2, and c3 are weighting coefficients.

        Example usage:
        ```python
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        import torch

        # Load the image and convert to tensor
        image_path = '/path/to/your/image.png'
        img = Image.open(image_path).convert('RGB')
        img_tensor = to_tensor(img).cuda().unsqueeze(0)  # Convert to tensor and move to GPU if available

        # Calculate the UIQM metric
        uiqm_value = torch_uiqm(img_tensor)
        print(f"UIQM value: {uiqm_value}")
        ```
    """
    x *= 255
    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 10)
    print("uicm", uicm, "uism", uism, "uiconm", uiconm)
    uiqm = (0.0282 * uicm) + (0.2953 * uism) + (3.5753 * uiconm)
    return uiqm


if __name__ == '__main__':
    image = r'/root/autodl-tmp/NeRFStudio/renders/eval/eval_img_0000.png'
    img = Image.open(image).convert('RGB')
    img = to_tensor(img).cuda().unsqueeze(0)
    img = torch.cat((img, img), 0)
    print(img.shape)
    res = torch_uciqe(img)
    print("uciqe: ", res.shape, res, "origin", uciqe(image))
    res = torch_uiqm(img)
    print("uiqm: ", res.shape, res)
