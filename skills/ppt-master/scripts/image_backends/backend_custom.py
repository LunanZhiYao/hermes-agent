#!/usr/bin/env python3
"""
Custom local API image generation backend.

Uses a local API endpoint (default: http://10.0.20.130:8091/v1/images/generations)
to generate images via POST requests. Compatible with the Hermes custom image_gen
plugin's API format.

Configuration keys:
  CUSTOM_IMAGE_API_URL  (optional) Override the default API URL
  CUSTOM_IMAGE_MODEL    (optional) Model name (default: uses server default)

Dependencies:
  pip install requests Pillow
"""

import sys

if __name__ == "__main__" and any(arg in {"-h", "--help", "help"} for arg in sys.argv[1:]):
    print(__doc__)
    print('Use via: python3 skills/ppt-master/scripts/image_gen.py "prompt" --backend custom')
    raise SystemExit(0)

import base64
import os
import time
import threading

import requests

from image_backends.backend_common import (
    MAX_RETRIES,
    is_rate_limit_error,
    normalize_image_size,
    resolve_output_path,
    retry_delay,
    save_image_bytes,
)


DEFAULT_API_URL = "http://10.0.20.130:8091/v1/images/generations"

# Aspect ratio -> dimension mapping for the custom API
ASPECT_RATIO_TO_DIMS = {
    "1:1":  {"width": 1024, "height": 1024},
    "16:9": {"width": 1024, "height": 512},
    "9:16": {"width": 512,  "height": 1024},
    "3:2":  {"width": 1024, "height": 683},
    "2:3":  {"width": 683,  "height": 1024},
    "4:3":  {"width": 1024, "height": 768},
    "3:4":  {"width": 768,  "height": 1024},
    "4:5":  {"width": 819,  "height": 1024},
    "5:4":  {"width": 1024, "height": 819},
    "21:9": {"width": 1024, "height": 439},
}

# image_size -> scale factor
IMAGE_SIZE_SCALE = {
    "512px": 0.5,
    "1K":    1.0,
    "2K":    2.0,
    "4K":    4.0,
}

VALID_ASPECT_RATIOS = list(ASPECT_RATIO_TO_DIMS.keys())


def _resolve_dims(aspect_ratio: str, image_size: str) -> dict:
    """Resolve aspect ratio and image size to pixel dimensions."""
    base = ASPECT_RATIO_TO_DIMS.get(aspect_ratio)
    if base is None:
        supported = list(VALID_ASPECT_RATIOS)
        raise ValueError(
            f"Unsupported aspect ratio '{aspect_ratio}' for custom backend. "
            f"Supported: {supported}"
        )
    scale = IMAGE_SIZE_SCALE.get(normalize_image_size(image_size), 1.0)
    width = max(16, round(base["width"] * scale / 16) * 16)
    height = max(16, round(base["height"] * scale / 16) * 16)
    return {"width": width, "height": height}


def _generate_image(prompt: str, aspect_ratio: str = "1:1",
                    image_size: str = "1K", output_dir: str = None,
                    filename: str = None, model: str = None,
                    api_url: str = None) -> str:
    """
    Image generation via custom local API endpoint.

    Returns:
        Path of the saved image file

    Raises:
        RuntimeError: When generation fails
    """
    if api_url is None:
        api_url = os.environ.get("CUSTOM_IMAGE_API_URL", DEFAULT_API_URL)

    dims = _resolve_dims(aspect_ratio, image_size)

    payload = {
        "model": model or os.environ.get("CUSTOM_IMAGE_MODEL") or None,
        "prompt": prompt,
        "negative_prompt": None,
        "n": 1,
        "height": dims["height"],
        "width": dims["width"],
        "response_format": "b64_json",
        "num_inference_steps": 9,
        "guidance_scale": 0.0,
        "seed": None,
        "cfg_normalization": False,
        "cfg_truncation": 1.0,
        "max_sequence_length": 512,
    }

    mode_label = f"Custom API: {api_url}"
    print(f"[Custom - {mode_label}]")
    print(f"  Model:        {payload['model'] or '(server default)'}")
    print(f"  Prompt:       {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"  Size:         {dims['width']}x{dims['height']} (from aspect_ratio={aspect_ratio}, image_size={image_size})")
    print()

    start_time = time.time()
    print(f"  [..] Generating...", end="", flush=True)

    # Heartbeat thread
    heartbeat_stop = threading.Event()

    def _heartbeat():
        while not heartbeat_stop.is_set():
            heartbeat_stop.wait(5)
            if not heartbeat_stop.is_set():
                elapsed = time.time() - start_time
                print(f" {elapsed:.0f}s...", end="", flush=True)

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to custom API at {api_url}: {exc}"
        ) from exc
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request timed out after 120 seconds for {api_url}")
    finally:
        heartbeat_stop.set()
        hb_thread.join(timeout=1)

    elapsed = time.time() - start_time
    print(f"\n  [DONE] Response received ({elapsed:.1f}s)")

    if response.status_code != 200:
        raise RuntimeError(
            f"API returned status code {response.status_code}: {response.text[:500]}"
        )

    result = response.json()
    data = result.get("data", [])
    if not data:
        raise RuntimeError("API returned no image data in response")

    first_image = data[0]
    b64_json = first_image.get("b64_json")
    if not b64_json:
        raise RuntimeError("API response does not contain base64 image data")

    image_data = base64.b64decode(b64_json)
    path = resolve_output_path(prompt, output_dir, filename, ".png")
    return save_image_bytes(image_data, path)


def generate(prompt: str,
             aspect_ratio: str = "1:1", image_size: str = "1K",
             output_dir: str = None, filename: str = None,
             model: str = None, max_retries: int = MAX_RETRIES) -> str:
    """
    Custom API image generation with automatic retry.

    Reads configuration from the current process environment or a .env file:
      CUSTOM_IMAGE_API_URL  (optional, default: http://10.0.20.130:8091/v1/images/generations)
      CUSTOM_IMAGE_MODEL    (optional)

    Args:
        prompt: Prompt text
        aspect_ratio: Aspect ratio
        image_size: Image size (512px, 1K, 2K, 4K)
        output_dir: Output directory
        filename: Output filename (without extension)
        model: Model name (default: server default)
        max_retries: Maximum number of retries

    Returns:
        Path of the saved image file
    """
    image_size = normalize_image_size(image_size)

    if aspect_ratio not in VALID_ASPECT_RATIOS:
        supported = list(VALID_ASPECT_RATIOS)
        raise ValueError(
            f"Unsupported aspect ratio '{aspect_ratio}' for custom backend. "
            f"Supported: {supported}"
        )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return _generate_image(prompt, aspect_ratio, image_size,
                                   output_dir, filename, model)
        except Exception as e:
            last_error = e
            if attempt < max_retries and is_rate_limit_error(e):
                delay = retry_delay(attempt, rate_limited=True)
                print(f"\n  [WARN] Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                      f"Waiting {delay}s before retry...")
                time.sleep(delay)
            elif attempt < max_retries:
                delay = retry_delay(attempt, rate_limited=False)
                print(f"\n  [WARN] Error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                      f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                break

    raise RuntimeError(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")
