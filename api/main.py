"""
FastAPI REST API for the Stable Diffusion Image Pipeline.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.sd_pipeline import SDPipeline, GenerationConfig
from pipeline.prompt_engineer import PromptEngineer
from pipeline.clip_scorer import CLIPScorer

logger = logging.getLogger(__name__)

# ── Global state (loaded once at startup) ─────────────────────────────────

pipeline: Optional[SDPipeline] = None
scorer: Optional[CLIPScorer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, scorer
    logger.info("Loading SD pipeline...")
    try:
        pipeline = SDPipeline(
            enable_cpu_offload=os.getenv("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
        )
        scorer = CLIPScorer()
    except Exception as e:
        logger.error("Pipeline load failed: %s", e)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Stable Diffusion 3.5 Image Pipeline API",
    description="Production-grade image generation API with CLIP quality scoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    style: str = Field("photorealistic", pattern="^(photorealistic|digital_art|anime|oil_painting|cinematic|minimal)$")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(28, ge=10, le=50)
    width: int = Field(1024, ge=512, le=1536)
    height: int = Field(1024, ge=512, le=1536)
    seed: Optional[int] = None
    min_clip_score: float = Field(0.0, ge=0.0, le=100.0)


class GenerateResponse(BaseModel):
    image_b64: str
    clip_score: float
    latency_ms: float
    prompt_used: str
    seed_used: Optional[int]


class BatchRequest(BaseModel):
    prompts: list[str] = Field(..., min_length=1, max_length=8)
    style: str = "photorealistic"
    num_inference_steps: int = Field(20, ge=10, le=50)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "scorer_loaded": scorer is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    # Build optimized prompt
    prompt, negative = PromptEngineer.build(req.prompt, style=req.style)

    config = GenerationConfig(
        prompt=prompt,
        negative_prompt=negative,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
        width=req.width,
        height=req.height,
        seed=req.seed,
    )

    result = pipeline.generate(config)
    image = result.images[0]

    # CLIP scoring
    clip_score = 0.0
    if scorer:
        clip_score = scorer.score(image, req.prompt)
        if req.min_clip_score > 0 and clip_score < req.min_clip_score:
            raise HTTPException(
                status_code=422,
                detail=f"Generated image CLIP score {clip_score:.1f} below threshold {req.min_clip_score}",
            )

    # Encode to base64 PNG
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return GenerateResponse(
        image_b64=img_b64,
        clip_score=clip_score,
        latency_ms=result.latency_ms,
        prompt_used=prompt,
        seed_used=req.seed,
    )


@app.post("/batch")
async def batch_generate(req: BatchRequest) -> list[dict]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    results = []
    for user_prompt in req.prompts:
        prompt, negative = PromptEngineer.build(user_prompt, style=req.style)
        config = GenerationConfig(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=req.num_inference_steps,
        )
        result = pipeline.generate(config)
        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        clip = scorer.score(image, user_prompt) if scorer else 0.0
        results.append({
            "prompt": user_prompt,
            "image_b64": base64.b64encode(buf.getvalue()).decode(),
            "clip_score": clip,
            "latency_ms": result.latency_ms,
        })
    return results


@app.get("/styles")
async def list_styles() -> list[str]:
    return list(PromptEngineer.__module__ and [
        "photorealistic", "digital_art", "anime",
        "oil_painting", "cinematic", "minimal"
    ])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
