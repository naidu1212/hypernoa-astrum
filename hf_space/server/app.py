"""FastAPI server for Hypernoa Astrum (OpenEnv compatible)."""

from openenv.core.env_server.http_server import create_app

from ..models import AstrumAction, AstrumObservation
from .astrum_environment import AstrumEnvironment

app = create_app(
    AstrumEnvironment,
    AstrumAction,
    AstrumObservation,
    env_name="hypernoa_astrum",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
