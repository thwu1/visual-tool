import colorsys
import math

import torch
from torch.nn import functional as F


def logprobs_from_logits(logits, labels, gather=True):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy


def color_map(value, key: str):
    saturation = 0.5  # Full saturation
    brightness = 1  # Full brightness

    if key.startswith("diff"):
        normalized_value = 1 / (1 + math.exp(-value))
        if key.endswith("logprob") or key.endswith("entropy") or key.endswith("prob"):
            hue, saturation, brightness = blue_to_white_to_red(normalized_value)
        else:
            raise ValueError("key should ends with 'logprob' or 'prob 'or 'entropy'")
    else:
        if key.endswith("prob"):
            # Assuming value is between 0 and 1
            hue = value * 120  # Green to Red
        elif key.endswith("entropy"):
            # Assuming value is between 0 and log(25000)
            hue = max((1 - (value / 2)) * 120, 0)
        else:
            raise ValueError("key should ends with 'prob' or 'entropy'")

    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, brightness)
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"


def blue_to_white_to_red(transition):
    """
    Calculate RGB values for a transition from blue to white to red.

    :param transition: A float between 0 and 1, where 0 is blue, 0.5 is white, and 1 is red.
    :return: A tuple (r, g, b) representing the RGB values.
    """
    saturation = 1
    brightness = 1  # Full brightness

    if transition <= 0.5:
        # From blue (240 degrees) to white
        hue = 240  # Blue
        saturation = (0.5-transition) * 2  # Decrease saturation to 0 at the midpoint (white)
    else:
        # From white to red (0 degrees)
        hue = 0  # Red
        saturation = (transition - 0.5) * 2  # Increase saturation from 0 to 1
    return hue, saturation, brightness


def openchat_template(text: str) -> str:
    """Template for OpenChat
    There shouldn't be spaces at the end of the string, otherwise the model will generate list, e.g.:
    1. ..., 2. ..., 3. ..."""
    return "GPT4 Correct User: " + text + "<|end_of_turn|> GPT4 Correct Assistant:"
