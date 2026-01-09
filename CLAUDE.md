# CLAUDE.md - Project Instructions

## Project Overview

GPX to STL Converter - A Streamlit web application that converts GPS track files (GPX) into 3D-printable elevation profile models (STL).

## Quick Commands

```bash
# Install dependencies
uv sync

# Run the Streamlit app
uv run streamlit run app.py

# Run CLI conversion
uv run python gps_stl.py ride.gpx -o output.stl --layout map

# Run tests
uv run pytest

# Lint code
uv run ruff check .
```

## Project Structure

- `app.py` - Streamlit web application UI
- `gps_stl.py` - Core GPX parsing and STL mesh generation logic
- `examples/` - Example GPX files and screenshots
- `pyproject.toml` - Project dependencies (requires Python 3.14+)

## Key Technical Details

### Layout Modes
- **Map Layout**: Creates ribbon-style mesh following actual GPS track shape using equirectangular projection
- **Linear Layout**: Creates straight elevation profile (distance vs elevation)

### Mesh Generation
- Uses numpy-stl for STL file creation
- Base plate with optional 45-degree beveled edges
- Text embossing on all 4 bevel sides using matplotlib TextPath
- Perpendicular offset calculation for ribbon width on map layout

### Important Functions in gps_stl.py
- `create_map_elevation_mesh()` - Main mesh generator for map layout with bevel and text
- `create_elevation_mesh()` - Linear profile mesh generator
- `create_text_mesh()` - Generates 3D embossed text for bevel sides
- `haversine_distance()` - Calculates geographic distance between points
- `resample_points()` - Resamples track to specified number of points

### Scaling Logic
When bevel is enabled, the track is scaled to fit within the effective area:
- `effective_width = width_mm - 2 * bevel_height_mm`
- `effective_depth = depth_mm - 2 * bevel_height_mm`

## Default Settings (Golden, CO example)
- Width: 170mm, Depth: 300mm
- Base Height: 6mm, Bevel: 5mm
- Text: "GOLDEN CO"
- Vertical Exaggeration: 3x
- Ribbon Width: 3mm
