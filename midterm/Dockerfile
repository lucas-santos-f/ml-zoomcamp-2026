FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY prep.py predict.py model.bin ./

EXPOSE 9696
ENTRYPOINT ["uv", "run", "--no-dev", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]