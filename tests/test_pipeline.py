"""
Tests for SD Pipeline — prompt engineering and CLIP scorer (pipeline mocked).
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


class TestPromptEngineer:
    def test_build_adds_style_suffix(self):
        from pipeline.prompt_engineer import PromptEngineer
        pos, neg = PromptEngineer.build("a cat", style="photorealistic")
        assert "hyperrealistic" in pos.lower()
        assert "cat" in pos

    def test_build_negative_contains_base(self):
        from pipeline.prompt_engineer import PromptEngineer
        _, neg = PromptEngineer.build("a dog", style="photorealistic")
        assert "blurry" in neg
        assert "watermark" in neg

    def test_extract_style_from_prompt(self):
        from pipeline.prompt_engineer import PromptEngineer
        prompt = "a mountain landscape --style anime"
        cleaned, style = PromptEngineer.extract_style(prompt)
        assert style == "anime"
        assert "--style" not in cleaned

    def test_extract_style_default(self):
        from pipeline.prompt_engineer import PromptEngineer
        cleaned, style = PromptEngineer.extract_style("hello world")
        assert style == "photorealistic"

    def test_batch_prompts(self):
        from pipeline.prompt_engineer import PromptEngineer
        results = PromptEngineer.batch_prompts(["cat", "dog", "bird"])
        assert len(results) == 3
        for pos, neg in results:
            assert isinstance(pos, str) and isinstance(neg, str)


class TestSDPipeline:
    @patch("pipeline.sd_pipeline.StableDiffusion3Pipeline")
    @patch("pipeline.sd_pipeline.torch")
    def test_generate_returns_result(self, mock_torch, mock_sd_class):
        from pipeline.sd_pipeline import SDPipeline, GenerationConfig

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.bfloat16 = "bfloat16"

        mock_pipe = MagicMock()
        fake_image = Image.new("RGB", (64, 64), color="red")
        mock_pipe.return_value.images = [fake_image]
        mock_sd_class.from_pretrained.return_value = mock_pipe.return_value

        pipeline = SDPipeline(model_id="test/model")
        config = GenerationConfig(prompt="a red square", num_inference_steps=1)
        result = pipeline.generate(config)

        assert len(result.images) == 1
        assert result.latency_ms > 0

    @patch("pipeline.sd_pipeline.StableDiffusion3Pipeline")
    @patch("pipeline.sd_pipeline.torch")
    def test_resolve_device_cpu(self, mock_torch, mock_sd_class):
        from pipeline.sd_pipeline import SDPipeline

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.bfloat16 = "bfloat16"
        mock_sd_class.from_pretrained.return_value = MagicMock()

        p = SDPipeline(model_id="test/model")
        assert p.device == "cpu"


class TestCLIPScorer:
    @patch("pipeline.clip_scorer.CLIPModel")
    @patch("pipeline.clip_scorer.CLIPProcessor")
    @patch("pipeline.clip_scorer.torch")
    def test_score_returns_float(self, mock_torch, mock_proc_cls, mock_model_cls):
        from pipeline.clip_scorer import CLIPScorer

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        import torch as real_torch
        mock_torch.no_grad.return_value.__enter__ = lambda s: s
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.sigmoid.return_value.item.return_value = 0.324

        mock_outputs = MagicMock()
        mock_outputs.logits_per_image = MagicMock()
        mock_model.return_value = mock_outputs

        mock_proc_cls.from_pretrained.return_value = MagicMock(
            return_value={"input_ids": MagicMock(), "pixel_values": MagicMock()}
        )

        scorer = CLIPScorer()
        img = Image.new("RGB", (224, 224))
        score = scorer.score(img, "a red square")
        assert isinstance(score, float)
