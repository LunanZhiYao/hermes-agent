"""Custom image generation backend.

Uses a local API endpoint at http://10.0.20.130:8091/v1/images/generations
to generate images via POST requests. Returns image URLs directly from the
API response.

Configuration (optional):
    - CUSTOM_IMAGE_API_URL: Override the default API URL
    - CUSTOM_IMAGE_DEFAULT_MODEL: Set a default model name (if needed)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

# Default API configuration
DEFAULT_API_URL = "http://10.0.20.130:8091/v1/images/generations"

# Aspect ratio to dimension mapping
_SIZES = {
    "landscape": {"width": 1024, "height": 512},
    "square": {"width": 512, "height": 512},
    "portrait": {"width": 512, "height": 1024},
}


def _get_api_url() -> str:
    """Get the API URL from environment or use default."""
    return os.environ.get("CUSTOM_IMAGE_API_URL", DEFAULT_API_URL)


class CustomImageGenProvider(ImageGenProvider):
    """Custom image generation provider using local API endpoint."""

    @property
    def name(self) -> str:
        return "custom"

    @property
    def display_name(self) -> str:
        return "Custom API"

    def is_available(self) -> bool:
        """Check if the custom API is reachable."""
        try:
            api_url = _get_api_url()
            # Try a simple health check by checking if the base URL is accessible
            base_url = api_url.replace("/v1/images/generations", "")
            response = requests.get(base_url, timeout=2)
            return response.status_code < 500
        except Exception:
            # Even if we can't reach it, still mark as available
            # The actual error will be caught during generation
            return True

    def list_models(self) -> List[Dict[str, Any]]:
        """Return available models for this provider."""
        return [
            {
                "id": "custom-model",
                "display": "Custom Model",
                "speed": "~5-10s",
                "strengths": "Local deployment, fast inference",
                "price": "Free",
            }
        ]

    def default_model(self) -> Optional[str]:
        return "custom-model"

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return setup metadata for the tools picker."""
        return {
            "name": "Custom API",
            "badge": "local",
            "tag": "Local image generation API endpoint",
            "env_vars": [
                {
                    "key": "CUSTOM_IMAGE_API_URL",
                    "prompt": "Custom Image API URL (optional)",
                    "url": "",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        size: Optional[str] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image using the custom API endpoint.
        
        Args:
            prompt: The text prompt describing the image to generate
            aspect_ratio: One of 'landscape', 'square', or 'portrait'
            size: Image size in format 'widthxheight' (e.g., '1024x1024'), overrides aspect_ratio if provided
            n: Number of images to generate (range: 1-4, default: 1)
            **kwargs: Additional parameters (ignored for forward compatibility)
            
        Returns:
            Dict containing the generation result
        """
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="custom",
                aspect_ratio=aspect,
            )

        # Validate and clamp n to valid range (1-4)
        n = max(1, min(4, int(n)))
        
        api_url = _get_api_url()
        
        # Determine dimensions: size parameter takes precedence over aspect_ratio
        if size:
            try:
                width, height = map(int, size.split('x'))
                dimensions = {"width": width, "height": height}
            except (ValueError, AttributeError):
                logger.warning(f"Invalid size format '{size}', falling back to aspect_ratio")
                dimensions = _SIZES.get(aspect, _SIZES["landscape"])
        else:
            dimensions = _SIZES.get(aspect, _SIZES["landscape"])

        # Build the request payload according to the API specification
        payload = {
            "model": None,  # Use default model on server side
            "prompt": prompt,
            "negative_prompt": None,  # No negative prompt by default
            "n": n,  # Number of images to generate (range: 1-4)
            "height": dimensions["height"],
            "width": dimensions["width"],
            "response_format": "b64_json",  # Return base64 encoded image
            "num_inference_steps": 30,  # Balanced quality/speed (range: 1-100)
            "guidance_scale": 7.5,  # Good balance for prompt adherence (range: 0.0-20.0)
            "seed": None,  # Random seed for variety
            "cfg_normalization": True,  # Enable CFG normalization for better quality
            "cfg_truncation": 1.0,  # Default truncation value (range: 0.0-2.0)
            "max_sequence_length": 512,  # Maximum sequence length (range: 1-512)
        }

        try:
            logger.debug(f"Sending image generation request to {api_url}")
            logger.debug(f"Payload: {payload}")
            
            response = requests.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,  # 2 minutes timeout for image generation
            )
            
            logger.debug(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                return error_response(
                    error=f"API returned status code {response.status_code}: {response.text}",
                    error_type="api_error",
                    provider="custom",
                    model="custom-model",
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            result = response.json()
            logger.debug(f"Response data: {result}")

            # Parse the response format: {"created": ..., "data": [{"b64_json": "..."}]}
            data = result.get("data", [])
            if not data:
                return error_response(
                    error="API returned no image data in response",
                    error_type="empty_response",
                    provider="custom",
                    model="custom-model",
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            first_image = data[0]
            b64_json = first_image.get("b64_json")
            
            if not b64_json:
                return error_response(
                    error="API response does not contain base64 image data",
                    error_type="empty_response",
                    provider="custom",
                    model="custom-model",
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            # Convert base64 to data URL format
            image_url = f"data:image/png;base64,{b64_json}"

            return success_response(
                image=image_url,
                model="custom-model",
                prompt=prompt,
                aspect_ratio=aspect,
                provider="custom",
                extra={
                    "size": f"{dimensions['width']}x{dimensions['height']}",
                    "api_url": api_url,
                    "format": "base64",
                },
            )

        except requests.exceptions.Timeout:
            return error_response(
                error="Request timed out after 120 seconds",
                error_type="timeout",
                provider="custom",
                model="custom-model",
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.exceptions.ConnectionError as exc:
            return error_response(
                error=f"Could not connect to API at {api_url}: {exc}",
                error_type="connection_error",
                provider="custom",
                model="custom-model",
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.error("Custom image generation failed", exc_info=True)
            return error_response(
                error=f"Image generation failed: {exc}",
                error_type="provider_error",
                provider="custom",
                model="custom-model",
                prompt=prompt,
                aspect_ratio=aspect,
            )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire CustomImageGenProvider into the registry."""
    ctx.register_image_gen_provider(CustomImageGenProvider())
