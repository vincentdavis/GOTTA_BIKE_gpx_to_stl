FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# Create and change to the app directory.
WORKDIR /app

# Copy local code to the container image.
COPY . .

# Install project dependencies.
RUN uv sync --frozen

# Run the app using the script.
CMD uv run streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.fileWatcherType none --browser.gatherUsageStats false --client.toolbarMode minimal
