"""FastAPI application for VietSignAI product inference."""

import time
from uuid import uuid4

from vietsign import __version__ as VIETSIGN_VERSION
from vietsign.inference.config import ApiConfig, load_api_config
from vietsign.inference.errors import InferenceError, PipelineUnavailableError
from vietsign.inference.model_manager import ModelManager
from vietsign.inference.orchestrator import VietSignPipeline
from vietsign.inference.pipelines import pipeline_infos
from vietsign.inference.schemas import (
    ErrorResponse,
    FeatureRequest,
    HealthResponse,
    RecognizeTranslateRequest,
    TranslationRequest,
)
from vietsign.inference.video import safe_upload_file
from vietsign.utils.logging import get_logger

LOGGER = get_logger(__name__)


def create_app(config: ApiConfig | None = None):
    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install the 'api' optional dependency to run the FastAPI application.") from error

    api_config = config or load_api_config("configs/api/default.yaml")
    manager = ModelManager(api_config)
    pipeline = VietSignPipeline(manager)
    app = FastAPI(title="VietSignAI API", version=VIETSIGN_VERSION)
    app.state.config = api_config
    app.state.manager = manager
    app.state.pipeline = pipeline
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.cors.allow_origins,
        allow_credentials=api_config.cors.allow_credentials,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(InferenceError)
    async def inference_error_handler(request: Request, error: InferenceError):
        request_id = getattr(request.state, "request_id", uuid4().hex)
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorResponse(
                request_id=request_id,
                error=error.code,
                message=error.message,
                details=error.details,
            ).model_dump(mode="json"),
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id", uuid4().hex)
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = request.state.request_id
        LOGGER.info(
            "request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        infos = pipeline_infos(manager)
        ready = [info.name for info in infos if info.ready]
        model_infos = manager.all_info()
        readiness = {
            "real_extractor_ready": False,
            "real_islr_checkpoint_ready": any(
                item.task == "isolated_recognition" and item.maturity != "development" for item in model_infos
            ),
            "real_cslr_checkpoint_ready": any(
                item.task == "continuous_recognition" and item.maturity != "development" for item in model_infos
            ),
            "real_translation_checkpoint_ready": any(
                item.task == "gloss_translation" and item.maturity != "development" for item in model_infos
            ),
        }
        return HealthResponse(
            status="healthy" if ready else "degraded",
            version=VIETSIGN_VERSION,
            device=manager.device.type,
            available_pipelines=ready,
            extractor_status="deferred_real_acceptance",
            loaded_models_count=manager.loaded_count(),
            real_vsl_ready=all(readiness.values()),
            readiness=readiness,
            warnings=["Phase 05 real-video acceptance deferred.", "Synthetic checkpoints are development-only."],
        )

    @app.get("/models")
    def models():
        return {"models": [item.model_dump(mode="json") for item in manager.all_info()]}

    @app.get("/pipelines")
    def pipelines():
        return {"pipelines": [item.model_dump(mode="json") for item in pipeline_infos(manager)]}

    @app.post("/translate")
    def translate(request: TranslationRequest, http_request: Request):
        gloss = request.glosses if request.glosses is not None else request.gloss
        if gloss is None:
            raise InferenceError("Provide gloss or glosses.", details={"field": "gloss"})
        return pipeline.translate_glosses(gloss, request_id=http_request.state.request_id, model_name=request.model)

    @app.post("/predict/continuous")
    def predict_continuous(request: FeatureRequest, http_request: Request):
        return pipeline.recognize_continuous(
            request.features_path,
            request_id=http_request.state.request_id,
            model_name=request.model,
        )

    @app.post("/recognize-and-translate")
    def recognize_translate(request: RecognizeTranslateRequest, http_request: Request):
        return pipeline.recognize_and_translate(
            request.features_path,
            request_id=http_request.state.request_id,
            cslr_model_name=request.cslr_model,
            translation_model_name=request.translation_model,
        )

    @app.post("/predict/isolated")
    async def predict_isolated_video(request: Request):
        form = await request.form()
        video = form.get("video")
        if video is None or not hasattr(video, "file"):
            raise InferenceError("Missing uploaded video.", details={"field": "video"})
        with safe_upload_file(video.file, video.filename or "upload.mp4", api_config.upload):
            raise PipelineUnavailableError("Isolated raw-video prediction requires real extractor/checkpoint readiness.")

    @app.post("/predict/continuous/upload")
    async def predict_continuous_upload(request: Request):
        form = await request.form()
        video = form.get("video")
        if video is None or not hasattr(video, "file"):
            raise InferenceError("Missing uploaded video.", details={"field": "video"})
        with safe_upload_file(video.file, video.filename or "upload.mp4", api_config.upload):
            raise PipelineUnavailableError("Raw-video continuous extraction awaits Phase 05 real acceptance.")

    return app


app = create_app()
