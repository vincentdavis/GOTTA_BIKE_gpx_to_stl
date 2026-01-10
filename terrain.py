#!/usr/bin/env python3
"""
Terrain data fetching and mesh generation for GPX to STL converter.
Supports USGS 3DEP (USA) and SRTM (global) elevation data.
"""

import math
from dataclasses import dataclass
import numpy as np

try:
    from stl import mesh
except ImportError:
    print("Installing numpy-stl...")
    import subprocess
    subprocess.check_call(["pip", "install", "numpy-stl", "--break-system-packages", "-q"])
    from stl import mesh


@dataclass
class TerrainGrid:
    """A grid of terrain elevation data."""
    elevations: np.ndarray  # 2D array of elevations (rows=lat, cols=lon)
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    resolution_m: float
    crs: str = "EPSG:4326"  # WGS84 by default

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def shape(self) -> tuple[int, int]:
        """Return (n_rows, n_cols) of the elevation grid."""
        return self.elevations.shape


def calculate_bounds_with_buffer(
    lats: list[float],
    lons: list[float],
    buffer_m: float = 500.0
) -> tuple[float, float, float, float]:
    """
    Calculate bounding box with buffer around coordinates.

    Args:
        lats: List of latitudes
        lons: List of longitudes
        buffer_m: Buffer distance in meters

    Returns:
        (min_lon, min_lat, max_lon, max_lat)
    """
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Convert buffer from meters to degrees (approximate)
    mid_lat = (min_lat + max_lat) / 2
    lat_deg_per_m = 1 / 111320
    lon_deg_per_m = 1 / (111320 * math.cos(math.radians(mid_lat)))

    lat_buffer = buffer_m * lat_deg_per_m
    lon_buffer = buffer_m * lon_deg_per_m

    return (
        min_lon - lon_buffer,
        min_lat - lat_buffer,
        max_lon + lon_buffer,
        max_lat + lat_buffer
    )


def fetch_terrain_usgs(
    bounds: tuple[float, float, float, float],
    resolution_m: float = 30.0
) -> TerrainGrid:
    """
    Fetch terrain data from USGS 3DEP service.

    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat) in WGS84
        resolution_m: Desired resolution in meters (10, 30, or 90)

    Returns:
        TerrainGrid with elevation data
    """
    try:
        import py3dep
    except ImportError:
        raise ImportError(
            "py3dep is required for USGS terrain data. "
            "Install with: pip install py3dep"
        )

    min_lon, min_lat, max_lon, max_lat = bounds

    # py3dep expects resolution in meters
    # Available resolutions: 10m, 30m, 60m for continental US
    valid_resolutions = [10, 30, 60]
    res = min(valid_resolutions, key=lambda x: abs(x - resolution_m))

    # Fetch DEM data
    dem = py3dep.get_dem(bounds, resolution=res)

    # Convert xarray DataArray to numpy
    elevations = dem.values

    # Handle NaN values (ocean, missing data) - set to minimum elevation
    if np.any(np.isnan(elevations)):
        valid_min = np.nanmin(elevations)
        elevations = np.nan_to_num(elevations, nan=valid_min)

    return TerrainGrid(
        elevations=elevations,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        resolution_m=res,
        crs="EPSG:4326"
    )


def fetch_terrain_srtm(
    bounds: tuple[float, float, float, float],
    resolution_m: float = 30.0
) -> TerrainGrid:
    """
    Fetch terrain data from SRTM (global coverage, ~30m resolution).
    Uses the elevation library or Open-Elevation API as fallback.

    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat) in WGS84
        resolution_m: Desired resolution in meters

    Returns:
        TerrainGrid with elevation data
    """
    import urllib.request
    import json

    min_lon, min_lat, max_lon, max_lat = bounds

    # Calculate grid dimensions based on resolution
    mid_lat = (min_lat + max_lat) / 2
    lat_extent = (max_lat - min_lat) * 111320
    lon_extent = (max_lon - min_lon) * 111320 * math.cos(math.radians(mid_lat))

    n_rows = max(2, int(lat_extent / resolution_m))
    n_cols = max(2, int(lon_extent / resolution_m))

    # Limit grid size to avoid overwhelming the API
    max_points = 1000
    if n_rows * n_cols > max_points:
        scale = math.sqrt(max_points / (n_rows * n_cols))
        n_rows = max(2, int(n_rows * scale))
        n_cols = max(2, int(n_cols * scale))

    # Generate grid points
    lats = np.linspace(max_lat, min_lat, n_rows)  # North to South
    lons = np.linspace(min_lon, max_lon, n_cols)

    # Create list of points for API query
    points = []
    for lat in lats:
        for lon in lons:
            points.append({"latitude": lat, "longitude": lon})

    # Query Open-Elevation API
    url = "https://api.open-elevation.com/api/v1/lookup"
    data = json.dumps({"locations": points}).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch elevation data: {e}")

    # Parse results into grid
    elevations = np.zeros((n_rows, n_cols))
    for i, point in enumerate(result['results']):
        row = i // n_cols
        col = i % n_cols
        elevations[row, col] = point['elevation']

    return TerrainGrid(
        elevations=elevations,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        resolution_m=resolution_m,
        crs="EPSG:4326"
    )


def fetch_terrain(
    bounds: tuple[float, float, float, float],
    resolution_m: float = 30.0,
    source: str = "auto"
) -> TerrainGrid:
    """
    Fetch terrain data from the best available source.

    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat) in WGS84
        resolution_m: Desired resolution in meters
        source: "usgs", "srtm", or "auto" (tries USGS first for USA)

    Returns:
        TerrainGrid with elevation data
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    # Check if bounds are within CONUS (Continental US)
    is_conus = (
        min_lat >= 24.5 and max_lat <= 49.5 and
        min_lon >= -125.0 and max_lon <= -66.5
    )

    if source == "auto":
        source = "usgs" if is_conus else "srtm"

    if source == "usgs":
        try:
            return fetch_terrain_usgs(bounds, resolution_m)
        except Exception as e:
            print(f"USGS fetch failed ({e}), falling back to SRTM...")
            return fetch_terrain_srtm(bounds, resolution_m)
    else:
        return fetch_terrain_srtm(bounds, resolution_m)


def terrain_grid_to_mesh(
    grid: TerrainGrid,
    width_mm: float = 150.0,
    depth_mm: float = 150.0,
    base_height_mm: float = 5.0,
    vertical_exaggeration: float = 2.0,
    min_terrain_height_mm: float = 2.0
) -> tuple[list, list]:
    """
    Convert a terrain grid to mesh vertices and faces.

    Args:
        grid: TerrainGrid with elevation data
        width_mm: Width of the model (X direction, corresponds to longitude)
        depth_mm: Depth of the model (Y direction, corresponds to latitude)
        base_height_mm: Height of the solid base
        vertical_exaggeration: Factor to exaggerate elevation
        min_terrain_height_mm: Minimum height above base for terrain

    Returns:
        Tuple of (vertices, faces) lists
    """
    elevations = grid.elevations
    n_rows, n_cols = elevations.shape

    # Normalize elevations
    min_elev = elevations.min()
    max_elev = elevations.max()
    elev_range = max_elev - min_elev

    if elev_range == 0:
        elev_range = 1  # Avoid division by zero

    # Scale elevations
    max_terrain_height_mm = 30.0
    normalized = (elevations - min_elev) / elev_range
    z_values = (
        base_height_mm +
        min_terrain_height_mm +
        normalized * max_terrain_height_mm * vertical_exaggeration / 2.0
    )

    # Generate X, Y coordinates for grid
    x_coords = np.linspace(0, width_mm, n_cols)
    y_coords = np.linspace(depth_mm, 0, n_rows)  # Flip Y so north is "up"

    vertices = []
    faces = []

    # Create terrain surface vertices
    # Vertex index = row * n_cols + col
    for row in range(n_rows):
        for col in range(n_cols):
            x = x_coords[col]
            y = y_coords[row]
            z = z_values[row, col]
            vertices.append((x, y, z))

    # Create terrain surface faces (2 triangles per grid cell)
    for row in range(n_rows - 1):
        for col in range(n_cols - 1):
            # Vertex indices for this cell
            tl = row * n_cols + col          # top-left
            tr = row * n_cols + col + 1      # top-right
            bl = (row + 1) * n_cols + col    # bottom-left
            br = (row + 1) * n_cols + col + 1  # bottom-right

            # Two triangles (CCW winding for outward normals)
            faces.append([tl, bl, tr])
            faces.append([tr, bl, br])

    # Add base plate vertices
    base_start_idx = len(vertices)

    # Bottom corners (z=0)
    vertices.append((0, 0, 0))                    # 0: front-left
    vertices.append((width_mm, 0, 0))             # 1: front-right
    vertices.append((width_mm, depth_mm, 0))      # 2: back-right
    vertices.append((0, depth_mm, 0))             # 3: back-left

    # Top corners (z=base_height_mm) - these connect to terrain edges
    vertices.append((0, 0, base_height_mm))               # 4
    vertices.append((width_mm, 0, base_height_mm))        # 5
    vertices.append((width_mm, depth_mm, base_height_mm)) # 6
    vertices.append((0, depth_mm, base_height_mm))        # 7

    b = base_start_idx

    # Bottom face
    faces.append([b+0, b+2, b+1])
    faces.append([b+0, b+3, b+2])

    # Front face (y=0)
    faces.append([b+0, b+1, b+5])
    faces.append([b+0, b+5, b+4])

    # Back face (y=depth_mm)
    faces.append([b+2, b+3, b+7])
    faces.append([b+2, b+7, b+6])

    # Left face (x=0)
    faces.append([b+0, b+4, b+7])
    faces.append([b+0, b+7, b+3])

    # Right face (x=width_mm)
    faces.append([b+1, b+2, b+6])
    faces.append([b+1, b+6, b+5])

    # Connect base plate top to terrain edges
    # Front edge (y=0, row=n_rows-1)
    front_row = n_rows - 1
    for col in range(n_cols - 1):
        terrain_left = front_row * n_cols + col
        terrain_right = front_row * n_cols + col + 1

        # Create vertical wall from base to terrain
        base_left_x = x_coords[col]
        base_right_x = x_coords[col + 1]

        # Add intermediate vertices for base top edge at this position
        base_left_idx = len(vertices)
        vertices.append((base_left_x, 0, base_height_mm))
        base_right_idx = len(vertices)
        vertices.append((base_right_x, 0, base_height_mm))

        # Wall face
        faces.append([base_left_idx, terrain_left, base_right_idx])
        faces.append([base_right_idx, terrain_left, terrain_right])

    # Back edge (y=depth_mm, row=0)
    for col in range(n_cols - 1):
        terrain_left = col
        terrain_right = col + 1

        base_left_x = x_coords[col]
        base_right_x = x_coords[col + 1]

        base_left_idx = len(vertices)
        vertices.append((base_left_x, depth_mm, base_height_mm))
        base_right_idx = len(vertices)
        vertices.append((base_right_x, depth_mm, base_height_mm))

        faces.append([base_left_idx, base_right_idx, terrain_left])
        faces.append([base_right_idx, terrain_right, terrain_left])

    # Left edge (x=0)
    for row in range(n_rows - 1):
        terrain_top = row * n_cols
        terrain_bottom = (row + 1) * n_cols

        base_top_y = y_coords[row]
        base_bottom_y = y_coords[row + 1]

        base_top_idx = len(vertices)
        vertices.append((0, base_top_y, base_height_mm))
        base_bottom_idx = len(vertices)
        vertices.append((0, base_bottom_y, base_height_mm))

        faces.append([base_top_idx, base_bottom_idx, terrain_top])
        faces.append([base_bottom_idx, terrain_bottom, terrain_top])

    # Right edge (x=width_mm)
    for row in range(n_rows - 1):
        terrain_top = row * n_cols + (n_cols - 1)
        terrain_bottom = (row + 1) * n_cols + (n_cols - 1)

        base_top_y = y_coords[row]
        base_bottom_y = y_coords[row + 1]

        base_top_idx = len(vertices)
        vertices.append((width_mm, base_top_y, base_height_mm))
        base_bottom_idx = len(vertices)
        vertices.append((width_mm, base_bottom_y, base_height_mm))

        faces.append([base_top_idx, terrain_top, base_bottom_idx])
        faces.append([base_bottom_idx, terrain_top, terrain_bottom])

    return vertices, faces


def create_route_on_terrain(
    grid: TerrainGrid,
    lats: list[float],
    lons: list[float],
    width_mm: float,
    depth_mm: float,
    base_height_mm: float,
    vertical_exaggeration: float,
    route_width_mm: float = 1.0,
    route_height_mm: float = 0.5
) -> tuple[list, list]:
    """
    Create a route ribbon on top of terrain.

    Args:
        grid: TerrainGrid with elevation data
        lats: Route latitudes
        lons: Route longitudes
        width_mm: Model width
        depth_mm: Model depth
        base_height_mm: Base height
        vertical_exaggeration: Vertical exaggeration factor
        route_width_mm: Width of the route ribbon
        route_height_mm: Height of route above terrain

    Returns:
        Tuple of (vertices, faces) for route ribbon
    """
    if len(lats) < 2:
        return [], []

    # Scale lat/lon to model coordinates
    lon_scale = width_mm / (grid.max_lon - grid.min_lon) if grid.max_lon != grid.min_lon else 1
    lat_scale = depth_mm / (grid.max_lat - grid.min_lat) if grid.max_lat != grid.min_lat else 1

    # Convert route points to model coordinates and sample terrain elevation
    x_coords = []
    y_coords = []
    z_coords = []

    elevations = grid.elevations
    n_rows, n_cols = elevations.shape
    min_elev = elevations.min()
    max_elev = elevations.max()
    elev_range = max_elev - min_elev if max_elev != min_elev else 1

    max_terrain_height_mm = 30.0
    min_terrain_height_mm = 2.0

    for lat, lon in zip(lats, lons):
        # Model coordinates
        # X: west (min_lon) → 0, east (max_lon) → width_mm
        # Y: south (min_lat) → 0, north (max_lat) → depth_mm (matches terrain grid)
        x = (lon - grid.min_lon) * lon_scale
        y = (lat - grid.min_lat) * lat_scale

        # Sample terrain elevation at this point (bilinear interpolation)
        # Grid coordinates
        col_f = (lon - grid.min_lon) / (grid.max_lon - grid.min_lon) * (n_cols - 1)
        row_f = (grid.max_lat - lat) / (grid.max_lat - grid.min_lat) * (n_rows - 1)

        col_f = max(0, min(n_cols - 1.001, col_f))
        row_f = max(0, min(n_rows - 1.001, row_f))

        col0, col1 = int(col_f), min(int(col_f) + 1, n_cols - 1)
        row0, row1 = int(row_f), min(int(row_f) + 1, n_rows - 1)

        # Bilinear interpolation
        t_col = col_f - col0
        t_row = row_f - row0

        elev = (
            elevations[row0, col0] * (1 - t_col) * (1 - t_row) +
            elevations[row0, col1] * t_col * (1 - t_row) +
            elevations[row1, col0] * (1 - t_col) * t_row +
            elevations[row1, col1] * t_col * t_row
        )

        # Scale elevation to model Z
        normalized = (elev - min_elev) / elev_range
        z = (
            base_height_mm +
            min_terrain_height_mm +
            normalized * max_terrain_height_mm * vertical_exaggeration / 2.0 +
            route_height_mm  # Raise above terrain
        )

        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

    n_points = len(x_coords)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)

    # Calculate perpendicular vectors for ribbon width
    dx = np.zeros(n_points)
    dy = np.zeros(n_points)

    for i in range(n_points):
        if i == 0:
            dx[i] = x_coords[1] - x_coords[0]
            dy[i] = y_coords[1] - y_coords[0]
        elif i == n_points - 1:
            dx[i] = x_coords[-1] - x_coords[-2]
            dy[i] = y_coords[-1] - y_coords[-2]
        else:
            dx[i] = x_coords[i + 1] - x_coords[i - 1]
            dy[i] = y_coords[i + 1] - y_coords[i - 1]

    lengths = np.sqrt(dx ** 2 + dy ** 2)
    lengths[lengths == 0] = 1
    dx = dx / lengths
    dy = dy / lengths

    # Perpendicular
    perp_x = -dy
    perp_y = dx

    half_width = route_width_mm / 2

    vertices = []
    faces = []

    # Create ribbon vertices (left and right edges)
    for i in range(n_points):
        left_x = x_coords[i] - perp_x[i] * half_width
        left_y = y_coords[i] - perp_y[i] * half_width
        right_x = x_coords[i] + perp_x[i] * half_width
        right_y = y_coords[i] + perp_y[i] * half_width
        z = z_coords[i]

        vertices.append((left_x, left_y, z))
        vertices.append((right_x, right_y, z))

    # Create faces (top surface of ribbon)
    for i in range(n_points - 1):
        left1 = i * 2
        right1 = i * 2 + 1
        left2 = (i + 1) * 2
        right2 = (i + 1) * 2 + 1

        faces.append([left1, left2, right1])
        faces.append([right1, left2, right2])

    return vertices, faces


def create_terrain_mesh_separate(
    grid: TerrainGrid,
    route_lats: list[float] = None,
    route_lons: list[float] = None,
    width_mm: float = 150.0,
    depth_mm: float = 150.0,
    base_height_mm: float = 5.0,
    vertical_exaggeration: float = 2.0,
    route_width_mm: float = 1.5,
    route_height_mm: float = 0.5
) -> tuple[mesh.Mesh, mesh.Mesh | None]:
    """
    Create terrain and route as separate meshes for visualization.

    Returns:
        Tuple of (terrain_mesh, route_mesh) - route_mesh is None if no route provided
    """
    # Generate terrain mesh
    terrain_verts, terrain_faces = terrain_grid_to_mesh(
        grid,
        width_mm=width_mm,
        depth_mm=depth_mm,
        base_height_mm=base_height_mm,
        vertical_exaggeration=vertical_exaggeration
    )

    # Create terrain mesh
    terrain_vertices = np.array(terrain_verts, dtype=np.float32)
    terrain_face_arr = np.array(terrain_faces, dtype=np.int32)
    terrain_mesh = mesh.Mesh(np.zeros(len(terrain_face_arr), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(terrain_face_arr):
        for j in range(3):
            terrain_mesh.vectors[i][j] = terrain_vertices[face[j]]

    # Create route mesh if provided
    route_mesh = None
    if route_lats is not None and route_lons is not None and len(route_lats) >= 2:
        route_verts, route_faces = create_route_on_terrain(
            grid,
            route_lats,
            route_lons,
            width_mm=width_mm,
            depth_mm=depth_mm,
            base_height_mm=base_height_mm,
            vertical_exaggeration=vertical_exaggeration,
            route_width_mm=route_width_mm,
            route_height_mm=route_height_mm
        )

        if route_verts:
            route_vertices = np.array(route_verts, dtype=np.float32)
            route_face_arr = np.array(route_faces, dtype=np.int32)
            route_mesh = mesh.Mesh(np.zeros(len(route_face_arr), dtype=mesh.Mesh.dtype))
            for i, face in enumerate(route_face_arr):
                for j in range(3):
                    route_mesh.vectors[i][j] = route_vertices[face[j]]

    return terrain_mesh, route_mesh


def create_terrain_mesh(
    grid: TerrainGrid,
    route_lats: list[float] = None,
    route_lons: list[float] = None,
    width_mm: float = 150.0,
    depth_mm: float = 150.0,
    base_height_mm: float = 5.0,
    vertical_exaggeration: float = 2.0,
    route_width_mm: float = 1.5,
    route_height_mm: float = 0.5
) -> mesh.Mesh:
    """
    Create a complete terrain mesh with optional route overlay.

    Args:
        grid: TerrainGrid with elevation data
        route_lats: Optional route latitudes for overlay
        route_lons: Optional route longitudes for overlay
        width_mm: Width of model
        depth_mm: Depth of model
        base_height_mm: Base height
        vertical_exaggeration: Vertical exaggeration factor
        route_width_mm: Width of route ribbon
        route_height_mm: Height of route above terrain

    Returns:
        numpy-stl Mesh object
    """
    # Generate terrain mesh
    terrain_verts, terrain_faces = terrain_grid_to_mesh(
        grid,
        width_mm=width_mm,
        depth_mm=depth_mm,
        base_height_mm=base_height_mm,
        vertical_exaggeration=vertical_exaggeration
    )

    all_vertices = list(terrain_verts)
    all_faces = list(terrain_faces)

    # Add route if provided
    if route_lats is not None and route_lons is not None and len(route_lats) >= 2:
        route_verts, route_faces = create_route_on_terrain(
            grid,
            route_lats,
            route_lons,
            width_mm=width_mm,
            depth_mm=depth_mm,
            base_height_mm=base_height_mm,
            vertical_exaggeration=vertical_exaggeration,
            route_width_mm=route_width_mm,
            route_height_mm=route_height_mm
        )

        if route_verts:
            route_base_idx = len(all_vertices)
            all_vertices.extend(route_verts)
            for face in route_faces:
                all_faces.append([f + route_base_idx for f in face])

    # Convert to numpy arrays
    vertices = np.array(all_vertices, dtype=np.float32)
    faces = np.array(all_faces, dtype=np.int32)

    # Create the mesh
    terrain_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = vertices[face[j]]

    return terrain_mesh
