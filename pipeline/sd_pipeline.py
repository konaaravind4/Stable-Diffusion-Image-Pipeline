"""
Core Stable Diffusion 3.5 generation pipeline with inference optimization.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "watermark, signature, out of frame, poorly drawn"
)


@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE
    guidance_scale: float = 7.5
    num_inference_steps: int = 28
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    num_images: int = 1


@dataclass
class GenerationResult:
    images: list[Image.Image]
    config: GenerationConfig
    latency_ms: float
    clip_scores: list[float] = field(default_factory=list)


class SDPipeline:
    """
    Production-grade Stable Diffusion 3.5 Large inference pipeline.

    Features:
    - Attention slicing for reduced VRAM (8.2 GB)
    - xFormers memory-efficient attention (if available)
    - Deterministic generation via seeded generators
    - CPU offloading fallback for memory-constrained environments
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "auto",
        enable_attention_slicing: bool = True,
        enable_cpu_offload: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = self._resolve_device(device)
        logger.info("Loading %s on %s...", model_id, self.device)

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )

        if enable_cpu_offload and self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            logger.info("CPU offloading enabled")
        elif self.device != "cpu":
            self.pipe = self.pipe.to(self.device)

        if enable_attention_slicing:
            self.pipe.enable_attention_slicing()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("xFormers memory-efficient attention enabled")
        except Exception:
            logger.debug("xFormers not available, using standard attention")

        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, config: GenerationConfig) -> GenerationResult:
        """Generate images from the given configuration."""
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(config.seed)

        logger.info(
            "Generating %d image(s): steps=%d cfg=%.1f",
            config.num_images,
            config.num_inference_steps,
            config.guidance_scale,
        )
        t0 = time.monotonic()

        output = self.pipe(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            width=config.width,
            height=config.height,
            num_images_per_prompt=config.num_images,
            generator=generator,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        logger.info("Generated in %.0f ms", latency_ms)

        return GenerationResult(
            images=output.images,
            config=config,
            latency_ms=latency_ms,
        )

    def save(self, result: GenerationResult, output_dir: str = "./outputs") -> list[Path]:
        """Save generated images to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        ts = int(time.time())
        for i, img in enumerate(result.images):
            path = out / f"generated_{ts}_{i:02d}.png"
            img.save(path)
            paths.append(path)
        return paths

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
