"""
tests/test_pipeline.py — Unit tests for SD pipeline (CPU-only, no model download).
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import io
import base64


# ─── Helper ────────────────────────────────────────────────────────────────────

def _make_dummy_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), color=(127, 0, 200))


def _b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ─── generator.py tests ────────────────────────────────────────────────────────

class TestImageToB64:
    def test_roundtrip(self):
        from pipeline.generator import _image_to_b64
        img = _make_dummy_image()
        b64 = _image_to_b64(img)
        recovered = _b64_to_image(b64)
        assert recovered.size == img.size


class TestGenerationConfig:
    def test_defaults(self):
        from pipeline.generator import GenerationConfig
        cfg = GenerationConfig(prompt="a cat")
        assert cfg.guidance_scale == 7.5
        assert cfg.num_inference_steps == 28
        assert cfg.seed is None


class TestSDPipeline:
    @patch("pipeline.generator.torch.cuda.is_available", return_value=False)
    @patch("pipeline.generator.torch.backends.mps.is_available", return_value=False)
    def test_generate_with_mocked_pipe(self, _mps, _cuda):
        from pipeline.generator import SDPipeline, GenerationConfig

        pipe = SDPipeline(model_id="fake/model")

        # Inject mocked diffusers pipe
        dummy_image = _make_dummy_image()
        mock_pipe = MagicMock()
        mock_pipe.return_value.images = [dummy_image]
        pipe._pipe = mock_pipe
        pipe._device = "cpu"

        cfg = GenerationConfig(prompt="a scenic mountain", seed=42)
        result = pipe.generate(cfg)

        assert result.seed == 42
        assert result.prompt == "a scenic mountain"
        assert result.latency_ms >= 0
        recovered = _b64_to_image(result.image_b64)
        assert recovered.size == (64, 64)

    def test_generate_raises_when_not_loaded(self):
        from pipeline.generator import SDPipeline, GenerationConfig
        pipe = SDPipeline()
        with pytest.raises(RuntimeError, match="not loaded"):
            pipe.generate(GenerationConfig(prompt="test"))


# ─── FastAPI endpoint tests ────────────────────────────────────────────────────

class TestAPIHealth:
    def test_health_returns_ok(self):
        from fastapi.testclient import TestClient
        import api.main as main_module

        # Inject a mock pipeline so lifespan is bypassed
        mock_pipe = MagicMock()
        dummy_image = _make_dummy_image()
        from pipeline.generator import GenerationResult
        mock_pipe.generate.return_value = GenerationResult(
            image_b64=_make_dummy_image_b64(),
            prompt="test",
            seed=1,
            latency_ms=100.0,
            width=64,
            height=64,
            steps=1,
            guidance_scale=7.5,
        )

        import pipeline.generator as gen_module
        gen_module._pipeline = mock_pipe

        client = TestClient(main_module.app, raise_server_exceptions=True)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def _make_dummy_image_b64() -> str:
    from pipeline.generator import _image_to_b64
    return _image_to_b64(_make_dummy_image())
