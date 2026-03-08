"""FastAPI server for Hypernoa Astrum (OpenEnv compatible)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import AstrumAction, AstrumObservation
from server.astrum_environment import AstrumEnvironment

_use_openenv = False
try:
    from openenv.core.env_server.http_server import create_app
    app = create_app(
        AstrumEnvironment,
        AstrumAction,
        AstrumObservation,
        env_name="hypernoa_astrum",
    )
    _use_openenv = True
except Exception:
    app = FastAPI(
        title="Hypernoa Astrum",
        description="Adaptive environment for training aligned intelligence",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _env = AstrumEnvironment()

    class ResetRequest(BaseModel):
        seed: Optional[int] = None
        episode_id: Optional[str] = None

    @app.get("/health")
    def health():
        return {"status": "healthy", "env": "hypernoa_astrum", "version": "0.1.0"}

    @app.post("/reset")
    def reset(req: ResetRequest | None = None):
        seed = req.seed if req else None
        episode_id = req.episode_id if req else None
        obs = _env.reset(seed=seed, episode_id=episode_id)
        return obs.model_dump()

    @app.post("/step")
    def step(action: AstrumAction):
        obs = _env.step(action)
        return obs.model_dump()


@app.get("/")
def root_info():
    return {
        "env": "hypernoa_astrum",
        "version": "0.1.0",
        "openenv": _use_openenv,
        "description": "Adaptive environment for training aligned intelligence",
        "endpoints": {
            "GET /": "This page",
            "GET /health": "Health check",
            "POST /reset": "Reset environment (optional: seed, episode_id)",
            "POST /step": "Take an action (action_type + params)",
        },
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
