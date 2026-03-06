"""
CLIP-based quality scoring for generated images.
Computes cosine similarity between text prompts and image embeddings.
"""
from __future__ import annotations

import logging
from typing import Union

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPScorer:
    """
    Scores generated images against their prompts using CLIP similarity.

    The score is the cosine similarity between the CLIP image embedding
    and the CLIP text embedding, scaled to [0, 100].
    """

    def __init__(self, model_id: str = CLIP_MODEL_ID, device: str = "auto"):
        self.device = self._resolve_device(device)
        logger.info("Loading CLIP scorer on %s...", self.device)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        logger.info("CLIP scorer ready.")

    def score(self, image: Image.Image, prompt: str) -> float:
        """Return a CLIP similarity score in [0, 100] for (image, prompt) pair."""
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # shape: [1, 1]
            score = torch.sigmoid(logits).item() * 100

        return round(score, 2)

    def score_batch(
        self, images: list[Image.Image], prompts: Union[list[str], str]
    ) -> list[float]:
        """Score a batch of images. If single prompt string, apply to all."""
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        scores = []
        for img, prompt in zip(images, prompts):
            scores.append(self.score(img, prompt))
        return scores

    def filter_by_score(
        self,
        images: list[Image.Image],
        prompts: Union[list[str], str],
        min_score: float = 25.0,
    ) -> list[tuple[Image.Image, float]]:
        """Return only (image, score) pairs that pass the minimum CLIP score threshold."""
        scores = self.score_batch(images, prompts)
        return [(img, s) for img, s in zip(images, scores) if s >= min_score]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
