"""
api/main.py — FastAPI REST service for the Stable Diffusion 3.5 pipeline.

Endpoints:
    POST /generate        — generate one image
    POST /generate/batch  — generate up to 4 images
    GET  /health          — health check
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from pipeline.generator import GenerationConfig, GenerationResult, init_pipeline, get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_id = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-3.5-large")
    logger.info("Initialising pipeline: %s", model_id)
    init_pipeline(model_id)
    yield
    logger.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stable Diffusion Image Pipeline",
    description="Production-grade SD 3.5 image generation REST API.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Request / Response Schemas ───────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = None
    steps: int = Field(28, ge=1, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    width: int = Field(1024, ge=256, le=1536)
    height: int = Field(1024, ge=256, le=1536)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    image_b64: str
    prompt: str
    seed: int
    latency_ms: float
    width: int
    height: int
    steps: int
    guidance_scale: float


class BatchGenerateRequest(BaseModel):
    prompts: list[str] = Field(..., min_length=1, max_length=4)
    steps: int = Field(28, ge=1, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    width: int = Field(1024, ge=256, le=1536)
    height: int = Field(1024, ge=256, le=1536)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("MODEL_ID", "stabilityai/stable-diffusion-3.5-large")}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    try:
        pipe = get_pipeline()
        cfg = GenerationConfig(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or GenerationConfig.__dataclass_fields__["negative_prompt"].default,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            seed=req.seed,
        )
        result = pipe.generate(cfg)
        return GenerateResponse(**result.__dict__)
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate/batch", response_model=list[GenerateResponse])
async def generate_batch(req: BatchGenerateRequest):
    try:
        pipe = get_pipeline()
        results = []
        for prompt in req.prompts:
            cfg = GenerationConfig(
                prompt=prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
            )
            results.append(GenerateResponse(**pipe.generate(cfg).__dict__))
        return results
    except Exception as exc:
        logger.exception("Batch generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
