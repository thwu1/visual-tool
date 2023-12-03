import torch
from torch.nn import functional as F
import colorsys

def logprobs_from_logits(logits, labels, gather=True):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def color_map(value):
    # Assuming value is between 0 and 1
    # Hue: 0 (red) to 120 (green) in HSV color space
    hue = value * 120  # Green to Red
    saturation = 0.5  # Full saturation
    brightness = 1  # Full brightness

    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, brightness)
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

def openchat_template(text: str) -> str:
    return "GPT4 Correct User: " + text + "<|end_of_turn|>GPT4 Correct Assistant:"