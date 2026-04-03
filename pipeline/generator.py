"""
pipeline/generator.py — Core Stable Diffusion 3.5 image generation pipeline.
"""

import io
import base64
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Parameters controlling the diffusion sampling process."""

    prompt: str
    negative_prompt: str = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
        "watermark, text, signature, jpeg artifacts"
    )
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Output from one generation call."""

    image_b64: str
    prompt: str
    seed: int
    latency_ms: float
    width: int
    height: int
    steps: int
    guidance_scale: float


def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class SDPipeline:
    """
    Wraps stabilityai/stable-diffusion-3.5-large (or any SD3 compatible model)
    with production-grade inference ergonomics.
    """

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-3.5-large"):
        self._model_id = model_id
        self._pipe = None
        self._device = "cpu"

    def load(self) -> None:
        """Load model weights. Call once at startup."""
        try:
            from diffusers import StableDiffusion3Pipeline  # type: ignore

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
            self._device = device

            logger.info("Loading SD pipeline from %s on %s …", self._model_id, device)
            self._pipe = StableDiffusion3Pipeline.from_pretrained(
                self._model_id, torch_dtype=dtype
            ).to(device)

            if device == "cuda":
                self._pipe.enable_attention_slicing()

            logger.info("Pipeline ready.")
        except Exception as exc:
            logger.error("Failed to load SD pipeline: %s", exc)
            raise

    def generate(self, cfg: GenerationConfig) -> GenerationResult:
        """Run one generation and return base64-encoded PNG."""
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        generator = None
        actual_seed = cfg.seed if cfg.seed is not None else int(time.time())
        generator = torch.Generator(device=self._device).manual_seed(actual_seed)

        t0 = time.perf_counter()
        result = self._pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            width=cfg.width,
            height=cfg.height,
            generator=generator,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        image: Image.Image = result.images[0]
        return GenerationResult(
            image_b64=_image_to_b64(image),
            prompt=cfg.prompt,
            seed=actual_seed,
            latency_ms=round(latency_ms, 1),
            width=cfg.width,
            height=cfg.height,
            steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
        )


# Module-level singleton — populated at FastAPI startup
_pipeline: Optional[SDPipeline] = None


def get_pipeline() -> SDPipeline:
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised.")
    return _pipeline


def init_pipeline(model_id: str) -> None:
    global _pipeline
    _pipeline = SDPipeline(model_id)
    _pipeline.load()
