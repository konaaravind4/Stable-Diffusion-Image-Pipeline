"""
Dynamic prompt engineering — builds optimized prompts for SD 3.5 generation.
"""
from __future__ import annotations

import re


STYLE_SUFFIXES = {
    "photorealistic": "hyperrealistic photography, 8k, professional lighting, DSLR, sharp focus",
    "digital_art": "digital art, trending on artstation, vibrant colors, concept art",
    "anime": "anime style, Studio Ghibli, detailed, soft lighting, 2D illustration",
    "oil_painting": "oil painting, textured brushstrokes, museum quality, rich colors",
    "cinematic": "cinematic shot, movie still, dramatic lighting, anamorphic lens",
    "minimal": "minimalist design, clean lines, flat illustration, simple background",
}

QUALITY_BOOSTER = (
    "masterpiece, best quality, highly detailed, sharp focus, professional"
)

NEGATIVE_ADDITIONS = {
    "photorealistic": "cartoon, illustration, painting, drawing, anime",
    "anime": "photorealistic, 3D render, ugly, poorly drawn",
    "cinematic": "amateur, snapshot, poorly lit, grainy",
}


class PromptEngineer:
    """Builds optimised positive/negative prompt pairs for consistent SD 3.5 output."""

    @staticmethod
    def build(
        user_prompt: str,
        style: str = "photorealistic",
        boost_quality: bool = True,
        extra_negative: str = "",
    ) -> tuple[str, str]:
        """
        Returns (positive_prompt, negative_prompt) tuple.
        """
        # Clean input
        cleaned = user_prompt.strip().rstrip(".,;:")

        # Assemble positive prompt
        parts = [cleaned]
        if style in STYLE_SUFFIXES:
            parts.append(STYLE_SUFFIXES[style])
        if boost_quality:
            parts.append(QUALITY_BOOSTER)
        positive = ", ".join(parts)

        # Assemble negative prompt
        base_negative = (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "watermark, signature, out of frame, poorly drawn, text, logo"
        )
        style_negative = NEGATIVE_ADDITIONS.get(style, "")
        negatives = [base_negative, style_negative, extra_negative]
        negative = ", ".join(n for n in negatives if n)

        return positive, negative

    @staticmethod
    def batch_prompts(prompts: list[str], style: str = "photorealistic") -> list[tuple[str, str]]:
        """Build prompt pairs for a list of user prompts."""
        return [PromptEngineer.build(p, style=style) for p in prompts]

    @staticmethod
    def extract_style(prompt: str) -> tuple[str, str]:
        """
        Detect style suffix in prompt (e.g. '--style anime') and strip it.
        Returns (cleaned_prompt, detected_style).
        """
        match = re.search(r"--style\s+(\w+)", prompt)
        if match:
            style = match.group(1).lower()
            cleaned = re.sub(r"--style\s+\w+", "", prompt).strip()
            return cleaned, style
        return prompt, "photorealistic"
