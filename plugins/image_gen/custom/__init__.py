"""Custom image generation backend.

Uses a local API endpoint at http://10.0.20.130:8091/v1/images/generations
to generate images via POST requests. Saves generated images to both the
Hermes cache directory and the user's workspace directory, returning the
workspace file path so the WebUI MEDIA: mechanism can render them inline.

Configuration (optional):
    - CUSTOM_IMAGE_API_URL: Override the default API URL
    - CUSTOM_IMAGE_DEFAULT_MODEL: Set a default model name (if needed)
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

# Default API configuration
DEFAULT_API_URL = "http://10.0.20.130:8091/v1/images/generations"

# Default dimensions when only aspect_ratio is specified (no explicit size)
_DEFAULT_SIZES = {
    "landscape": {"width": 1024, "height": 512},
    "square": {"width": 512, "height": 512},
    "portrait": {"width": 512, "height": 1024},
}

# Reasonable dimension limits for the custom API
_MAX_DIMENSION = 4096
_MIN_DIMENSION = 64


def _get_api_url() -> str:
    """Get the API URL from environment or use default."""
    return os.environ.get("CUSTOM_IMAGE_API_URL", DEFAULT_API_URL)


def _get_workspace_dir() -> Optional[Path]:
    """获取当前用户的 workspace 目录路径。

    多租户模式下，WebUI 会设置 HERMES_HOME 为租户根目录（如 ~/.hermes/users/xxx/），
    用户 workspace 固定在 $HERMES_HOME/workspace/。

    优先级：
    1. HERMES_HOME/workspace/（多租户模式下最可靠，由 WebUI streaming.py 设置）
    2. TERMINAL_CWD 环境变量（非多租户模式的回退）
    3. 返回 None 表示无法确定 workspace
    """
    # 优先：从 HERMES_HOME 推导 workspace（多租户模式下 HERMES_HOME 已被设为租户目录）
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home:
        ws = Path(hermes_home).expanduser().resolve() / "workspace"
        if ws.is_dir():
            logger.debug("通过 HERMES_HOME 确定用户 workspace: %s", ws)
            return ws
        # 目录不存在时尝试创建（租户首次生成图片的情况）
        try:
            ws.mkdir(parents=True, exist_ok=True)
            logger.debug("已创建用户 workspace 目录: %s", ws)
            return ws
        except Exception as exc:
            logger.debug("创建 workspace 目录失败: %s", exc)

    # 回退：TERMINAL_CWD（非多租户模式，或 HERMES_HOME 未设置时）
    terminal_cwd = os.environ.get("TERMINAL_CWD", "").strip()
    if terminal_cwd:
        ws = Path(terminal_cwd).expanduser().resolve()
        if ws.is_dir():
            logger.debug("通过 TERMINAL_CWD 确定 workspace: %s", ws)
            return ws

    return None


def _save_to_workspace(cached_path: Path, prompt: str) -> Optional[Path]:
    """将缓存中的图片复制到用户 workspace 根目录下。

    Args:
        cached_path: 图片在 Hermes cache 中的路径
        prompt: 生成图片的提示词（未使用，保留接口兼容）

    Returns:
        复制后的 workspace 文件路径，如果无法确定 workspace 则返回 None
    """
    workspace = _get_workspace_dir()
    if workspace is None:
        logger.debug("无法确定 workspace 目录，跳过复制到 workspace")
        return None

    # 直接保存到 workspace 根目录（不创建子目录）
    dest_path = workspace / cached_path.name
    try:
        shutil.copy2(cached_path, dest_path)
        logger.info("图片已复制到用户 workspace: %s", dest_path)
        return dest_path
    except Exception as exc:
        logger.warning("复制图片到 workspace 失败: %s", exc)
        return None


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
                # Clamp to valid range
                width = max(_MIN_DIMENSION, min(_MAX_DIMENSION, width))
                height = max(_MIN_DIMENSION, min(_MAX_DIMENSION, height))
                dimensions = {"width": width, "height": height}
            except (ValueError, AttributeError):
                logger.warning(f"Invalid size format '{size}', falling back to aspect_ratio")
                dimensions = _DEFAULT_SIZES.get(aspect, _DEFAULT_SIZES["landscape"])
        else:
            dimensions = _DEFAULT_SIZES.get(aspect, _DEFAULT_SIZES["landscape"])

        # Build the request payload according to the API specification
        payload = {
            "model": None,  # Use default model on server side
            "prompt": prompt,
            "negative_prompt": None,  # No negative prompt by default
            "n": n,  # Number of images to generate (range: 1-4)
            "height": dimensions["height"],
            "width": dimensions["width"],
            "response_format": "b64_json",  # Return base64 encoded image
            "num_inference_steps": 9,  # Balanced quality/speed (range: 1-100)
            "guidance_scale": 0.0,  # Good balance for prompt adherence (range: 0.0-20.0)
            "seed": None,  # Random seed for variety
            "cfg_normalization": False,  # Enable CFG normalization for better quality
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

            # 将 base64 数据保存为本地文件（而非拼接 data URL）
            # 1. 保存到 Hermes 缓存目录 ($HERMES_HOME/cache/images/)
            cached_path = save_b64_image(
                b64_json, prefix="custom", extension="png"
            )
            logger.info(f"图片已缓存到: {cached_path}")

            # 2. 复制到用户 workspace 目录，优先返回 workspace 路径
            #    这样前端 MEDIA: 机制可以通过 api/media 端点加载图片
            workspace_path = _save_to_workspace(cached_path, prompt)

            # 返回 workspace 路径（优先）或缓存路径作为 image 字段
            # Agent 会将此路径包装为 MEDIA:<path>，前端通过 api/media 渲染
            image_path = str(workspace_path) if workspace_path else str(cached_path)

            return success_response(
                image=image_path,
                model="custom-model",
                prompt=prompt,
                aspect_ratio=aspect,
                provider="custom",
                extra={
                    "size": f"{dimensions['width']}x{dimensions['height']}",
                    "api_url": api_url,
                    "format": "file",
                    "cached_path": str(cached_path),
                    "workspace_path": str(workspace_path) if workspace_path else None,
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
