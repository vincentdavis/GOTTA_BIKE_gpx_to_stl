#!/usr/bin/env python3
"""
GPX to STL Converter
Converts GPS track files to 3D-printable elevation profile models.
Similar to the 3D profiles shown on VeloViewer.
"""

import argparse
import math
from dataclasses import dataclass
import numpy as np

try:
    import gpxpy
    import gpxpy.gpx
except ImportError:
    print("Installing gpxpy...")
    import subprocess

    subprocess.check_call(["pip", "install", "gpxpy", "--break-system-packages", "-q"])
    import gpxpy
    import gpxpy.gpx

try:
    from stl import mesh
except ImportError:
    print("Installing numpy-stl...")
    import subprocess

    subprocess.check_call(["pip", "install", "numpy-stl", "--break-system-packages", "-q"])
    from stl import mesh


@dataclass
class TrackPoint:
    """A point along the track with distance and elevation."""
    distance: float  # cumulative distance in meters
    elevation: float  # elevation in meters
    lat: float
    lon: float


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance between two GPS coordinates in meters."""
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def parse_gpx(gpx_file: str) -> list[TrackPoint]:
    """Parse a GPX file and return a list of TrackPoints with cumulative distance."""
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)

    points = []
    cumulative_distance = 0.0
    prev_point = None

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.elevation is None:
                    continue

                if prev_point is not None:
                    cumulative_distance += haversine_distance(
                        prev_point.latitude, prev_point.longitude,
                        point.latitude, point.longitude
                    )

                points.append(TrackPoint(
                    distance=cumulative_distance,
                    elevation=point.elevation,
                    lat=point.latitude,
                    lon=point.longitude
                ))
                prev_point = point

    # Also check routes if no tracks
    if not points:
        for route in gpx.routes:
            for point in route.points:
                if point.elevation is None:
                    continue

                if prev_point is not None:
                    cumulative_distance += haversine_distance(
                        prev_point.latitude, prev_point.longitude,
                        point.latitude, point.longitude
                    )

                points.append(TrackPoint(
                    distance=cumulative_distance,
                    elevation=point.elevation,
                    lat=point.latitude,
                    lon=point.longitude
                ))
                prev_point = point

    return points


def smooth_elevations(points: list[TrackPoint], window_size: int = 5) -> list[TrackPoint]:
    """Apply a simple moving average to smooth elevation data."""
    if len(points) < window_size:
        return points

    elevations = [p.elevation for p in points]
    smoothed = []

    for i in range(len(elevations)):
        start = max(0, i - window_size // 2)
        end = min(len(elevations), i + window_size // 2 + 1)
        smoothed.append(sum(elevations[start:end]) / (end - start))

    return [
        TrackPoint(p.distance, smoothed[i], p.lat, p.lon)
        for i, p in enumerate(points)
    ]


def resample_points(points: list[TrackPoint], num_points: int = 500) -> list[TrackPoint]:
    """Resample track to a fixed number of evenly-spaced points."""
    if len(points) < 2:
        return points

    total_distance = points[-1].distance
    if total_distance == 0:
        return points

    resampled = []
    point_idx = 0

    for i in range(num_points):
        target_distance = (i / (num_points - 1)) * total_distance

        # Find surrounding points
        while point_idx < len(points) - 1 and points[point_idx + 1].distance < target_distance:
            point_idx += 1

        if point_idx >= len(points) - 1:
            resampled.append(points[-1])
            continue

        # Interpolate
        p1, p2 = points[point_idx], points[point_idx + 1]
        if p2.distance == p1.distance:
            t = 0
        else:
            t = (target_distance - p1.distance) / (p2.distance - p1.distance)

        resampled.append(TrackPoint(
            distance=target_distance,
            elevation=p1.elevation + t * (p2.elevation - p1.elevation),
            lat=p1.lat + t * (p2.lat - p1.lat),
            lon=p1.lon + t * (p2.lon - p1.lon)
        ))

    return resampled


def create_elevation_mesh(
        points: list[TrackPoint],
        width_mm: float = 150.0,
        depth_mm: float = 20.0,
        base_height_mm: float = 5.0,
        vertical_exaggeration: float = 2.0,
        min_elevation_height_mm: float = 2.0
) -> mesh.Mesh:
    """
    Create a 3D mesh from track points.

    Args:
        points: List of TrackPoints
        width_mm: Total width of the model in mm
        depth_mm: Depth (thickness) of the model in mm
        base_height_mm: Height of the flat base in mm
        vertical_exaggeration: Factor to exaggerate elevation differences
        min_elevation_height_mm: Minimum height above base for the profile

    Returns:
        numpy-stl Mesh object
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to create a mesh")

    # Normalize distances to width
    total_distance = points[-1].distance
    x_coords = np.array([p.distance / total_distance * width_mm for p in points])

    # Normalize elevations
    elevations = np.array([p.elevation for p in points])
    min_elev = elevations.min()
    max_elev = elevations.max()
    elev_range = max_elev - min_elev

    if elev_range == 0:
        elev_range = 1  # Avoid division by zero for flat routes

    # Scale elevations to reasonable height with exaggeration
    max_profile_height = 30.0  # mm, maximum height of the profile part
    normalized_elevations = (elevations - min_elev) / elev_range
    z_coords = base_height_mm + min_elevation_height_mm + \
               normalized_elevations * max_profile_height * vertical_exaggeration / 2.0

    n_points = len(points)

    # Create vertices for the mesh
    # We need: front face, back face, top face, bottom face, left cap, right cap

    # Front vertices (y = 0): bottom-left to top-left, along profile, top-right to bottom-right
    # Back vertices (y = depth): same pattern

    vertices = []
    faces = []

    # Front face vertices (y = 0)
    front_bottom = [(x, 0, 0) for x in x_coords]
    front_top = [(x_coords[i], 0, z_coords[i]) for i in range(n_points)]

    # Back face vertices (y = depth)
    back_bottom = [(x, depth_mm, 0) for x in x_coords]
    back_top = [(x_coords[i], depth_mm, z_coords[i]) for i in range(n_points)]

    # Build vertex list
    # 0 to n_points-1: front bottom
    # n_points to 2*n_points-1: front top
    # 2*n_points to 3*n_points-1: back bottom
    # 3*n_points to 4*n_points-1: back top

    vertices.extend(front_bottom)
    vertices.extend(front_top)
    vertices.extend(back_bottom)
    vertices.extend(back_top)

    # Front face triangles
    for i in range(n_points - 1):
        # Two triangles per quad
        faces.append([i, i + 1, i + n_points])  # bottom-left, bottom-right, top-left
        faces.append([i + 1, i + n_points + 1, i + n_points])  # bottom-right, top-right, top-left

    # Back face triangles (reversed winding for outward normals)
    offset = 2 * n_points
    for i in range(n_points - 1):
        faces.append([offset + i, offset + i + n_points, offset + i + 1])
        faces.append([offset + i + 1, offset + i + n_points, offset + i + n_points + 1])

    # Top face (profile surface)
    for i in range(n_points - 1):
        front_top_idx = n_points + i
        back_top_idx = 3 * n_points + i
        faces.append([front_top_idx, back_top_idx, front_top_idx + 1])
        faces.append([front_top_idx + 1, back_top_idx, back_top_idx + 1])

    # Bottom face
    for i in range(n_points - 1):
        front_bottom_idx = i
        back_bottom_idx = 2 * n_points + i
        faces.append([front_bottom_idx, front_bottom_idx + 1, back_bottom_idx])
        faces.append([front_bottom_idx + 1, back_bottom_idx + 1, back_bottom_idx])

    # Left cap (x = 0)
    faces.append([0, n_points, 2 * n_points])  # front-bottom, front-top, back-bottom
    faces.append([n_points, 3 * n_points, 2 * n_points])  # front-top, back-top, back-bottom

    # Right cap (x = width)
    last = n_points - 1
    faces.append([last, 2 * n_points + last, n_points + last])
    faces.append([n_points + last, 2 * n_points + last, 3 * n_points + last])

    # Convert to numpy arrays
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    # Create the mesh
    elevation_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            elevation_mesh.vectors[i][j] = vertices[face[j]]

    return elevation_mesh


def create_text_mesh_for_side(
        text: str,
        text_height_mm: float,
        text_depth_mm: float,
        bevel_height_mm: float,
        base_height_mm: float,
        side_length_mm: float,
        side: str,
        width_mm: float,
        depth_mm: float
) -> tuple[list, list]:
    """
    Create 3D text mesh vertices and faces for embossing on one bevel side.

    Args:
        side: One of 'front', 'back', 'left', 'right'

    Returns:
        Tuple of (vertices, faces) lists
    """
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    if not text:
        return [], []

    vertices = []
    faces = []

    # Create text path
    font_props = FontProperties(family='sans-serif', weight='bold')
    text_path = TextPath((0, 0), text, size=text_height_mm, prop=font_props)

    # Get bounding box to center the text
    bbox = text_path.get_extents()
    text_width = bbox.width
    text_actual_height = bbox.height

    # Center text horizontally on the side
    along_offset = (side_length_mm - text_width) / 2 - bbox.x0

    # Iterate through path vertices to create 3D geometry
    for path in text_path.to_polygons():
        if len(path) < 3:
            continue

        n_verts = len(path) - 1  # Last vertex is same as first
        base_idx = len(vertices)

        # Create front and back vertices for each point in the path
        for i in range(n_verts):
            along = path[i][0] + along_offset  # Position along the side
            up_2d = path[i][1]  # This is the "up" direction in 2D text

            # Map up_2d to position along the bevel (z and perpendicular)
            bevel_frac = (up_2d - bbox.y0) / text_actual_height if text_actual_height > 0 else 0.5

            # Position on the angled bevel surface
            perp_offset = bevel_frac * bevel_height_mm * 0.8
            z_3d = bevel_frac * base_height_mm * 0.8 + base_height_mm * 0.1

            # Calculate 3D position based on which side
            if side == 'front':
                # Front: y=0, text along x-axis
                x = along
                y_back = perp_offset
                y_front = perp_offset - text_depth_mm / math.sqrt(2)
                z_front = z_3d + text_depth_mm / math.sqrt(2)
                vertices.append((x, y_back, z_3d))
                vertices.append((x, y_front, z_front))
            elif side == 'back':
                # Back: y=depth, text along x-axis (mirrored)
                x = width_mm - along  # Mirror text
                y_back = depth_mm - perp_offset
                y_front = depth_mm - perp_offset + text_depth_mm / math.sqrt(2)
                z_front = z_3d + text_depth_mm / math.sqrt(2)
                vertices.append((x, y_back, z_3d))
                vertices.append((x, y_front, z_front))
            elif side == 'left':
                # Left: x=0, text along y-axis (mirrored to read from outside)
                y = depth_mm - along  # Mirror text
                x_back = perp_offset
                x_front = perp_offset - text_depth_mm / math.sqrt(2)
                z_front = z_3d + text_depth_mm / math.sqrt(2)
                vertices.append((x_back, y, z_3d))
                vertices.append((x_front, y, z_front))
            elif side == 'right':
                # Right: x=width, text along y-axis
                y = along
                x_back = width_mm - perp_offset
                x_front = width_mm - perp_offset + text_depth_mm / math.sqrt(2)
                z_front = z_3d + text_depth_mm / math.sqrt(2)
                vertices.append((x_back, y, z_3d))
                vertices.append((x_front, y, z_front))

        # Create faces for this polygon
        # Front face (using triangulation - simple fan from center)
        center_front = [0.0, 0.0, 0.0]
        for i in range(n_verts):
            front_idx = base_idx + i * 2 + 1
            center_front[0] += vertices[front_idx][0]
            center_front[1] += vertices[front_idx][1]
            center_front[2] += vertices[front_idx][2]
        center_front = [c / n_verts for c in center_front]

        center_idx = len(vertices)
        vertices.append(tuple(center_front))

        # Front face triangles (fan triangulation)
        for i in range(n_verts):
            front_idx1 = base_idx + i * 2 + 1
            front_idx2 = base_idx + ((i + 1) % n_verts) * 2 + 1
            faces.append([center_idx, front_idx1, front_idx2])

        # Back face (same but reversed winding)
        center_back = [0.0, 0.0, 0.0]
        for i in range(n_verts):
            back_idx = base_idx + i * 2
            center_back[0] += vertices[back_idx][0]
            center_back[1] += vertices[back_idx][1]
            center_back[2] += vertices[back_idx][2]
        center_back = [c / n_verts for c in center_back]

        center_back_idx = len(vertices)
        vertices.append(tuple(center_back))

        for i in range(n_verts):
            back_idx1 = base_idx + i * 2
            back_idx2 = base_idx + ((i + 1) % n_verts) * 2
            faces.append([center_back_idx, back_idx2, back_idx1])

        # Side faces (connecting front and back)
        for i in range(n_verts):
            back1 = base_idx + i * 2
            front1 = base_idx + i * 2 + 1
            back2 = base_idx + ((i + 1) % n_verts) * 2
            front2 = base_idx + ((i + 1) % n_verts) * 2 + 1

            faces.append([back1, front1, back2])
            faces.append([front1, front2, back2])

    return vertices, faces


def create_text_mesh(
        text: str,
        text_height_mm: float,
        text_depth_mm: float,
        bevel_height_mm: float,
        base_height_mm: float,
        width_mm: float,
        depth_mm: float
) -> tuple[list, list]:
    """
    Create 3D text mesh vertices and faces for embossing on all 4 bevel sides.

    Returns:
        Tuple of (vertices, faces) lists
    """
    if not text:
        return [], []

    all_vertices = []
    all_faces = []

    # Create text for each side
    sides = [
        ('front', width_mm),
        ('back', width_mm),
        ('left', depth_mm),
        ('right', depth_mm)
    ]

    for side, side_length in sides:
        verts, fcs = create_text_mesh_for_side(
            text, text_height_mm, text_depth_mm,
            bevel_height_mm, base_height_mm, side_length,
            side, width_mm, depth_mm
        )
        if verts:
            base_idx = len(all_vertices)
            all_vertices.extend(verts)
            for face in fcs:
                all_faces.append([f + base_idx for f in face])

    return all_vertices, all_faces


def create_map_elevation_mesh(
        points: list[TrackPoint],
        width_mm: float = 150.0,
        depth_mm: float = 20.0,
        base_height_mm: float = 5.0,
        vertical_exaggeration: float = 2.0,
        min_elevation_height_mm: float = 2.0,
        ribbon_width_mm: float = 10.0,
        bevel_height_mm: float = 0.0,
        bevel_text: str = "",
        bevel_text_height_mm: float = 5.0,
        bevel_text_depth_mm: float = 1.0
) -> mesh.Mesh:
    """
    Create a 3D mesh from track points following the actual GPS track shape.

    Args:
        points: List of TrackPoints with lat/lon coordinates
        width_mm: Maximum width of the model in mm (for scaling)
        depth_mm: Maximum depth of the model in mm (for scaling)
        base_height_mm: Height of the flat base in mm
        vertical_exaggeration: Factor to exaggerate elevation differences
        min_elevation_height_mm: Minimum height above base for the profile
        ribbon_width_mm: Width of the ribbon/path in mm
        bevel_height_mm: Height of beveled edge (0 for no bevel)
        bevel_text: Text to emboss on the front bevel
        bevel_text_height_mm: Height of the text
        bevel_text_depth_mm: Depth of text extrusion

    Returns:
        numpy-stl Mesh object
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to create a mesh")

    # Extract lat/lon coordinates
    lats = np.array([p.lat for p in points])
    lons = np.array([p.lon for p in points])

    # Calculate geographic bounds
    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()

    # Convert to approximate meters using equirectangular projection
    # Longitude degrees to meters varies with latitude
    mid_lat = (min_lat + max_lat) / 2
    lat_to_meters = 111320  # meters per degree latitude (approximately constant)
    lon_to_meters = 111320 * math.cos(math.radians(mid_lat))  # varies with latitude

    # Calculate geographic extent in meters
    lat_extent_m = (max_lat - min_lat) * lat_to_meters
    lon_extent_m = (max_lon - min_lon) * lon_to_meters

    # Handle edge cases (very small extent)
    if lat_extent_m < 1:
        lat_extent_m = 1
    if lon_extent_m < 1:
        lon_extent_m = 1

    # Calculate effective dimensions accounting for bevel inset
    # The track should fit within the top surface of the beveled base
    bevel = min(bevel_height_mm, base_height_mm, width_mm / 4, depth_mm / 4)
    effective_width = width_mm - 2 * bevel
    effective_depth = depth_mm - 2 * bevel

    # Scale to fit within effective area while preserving aspect ratio
    scale_x = effective_width / lon_extent_m if lon_extent_m > 0 else 1
    scale_y = effective_depth / lat_extent_m if lat_extent_m > 0 else 1
    scale = min(scale_x, scale_y)  # Use smaller scale to fit within bounds

    # Convert lat/lon to X/Y coordinates in mm, offset by bevel to center on top surface
    x_coords = (lons - min_lon) * lon_to_meters * scale + bevel
    y_coords = (lats - min_lat) * lat_to_meters * scale + bevel

    # Normalize elevations
    elevations = np.array([p.elevation for p in points])
    min_elev = elevations.min()
    max_elev = elevations.max()
    elev_range = max_elev - min_elev

    if elev_range == 0:
        elev_range = 1  # Avoid division by zero for flat routes

    # Scale elevations to reasonable height with exaggeration
    max_profile_height = 30.0  # mm
    normalized_elevations = (elevations - min_elev) / elev_range
    z_coords = base_height_mm + min_elevation_height_mm + \
               normalized_elevations * max_profile_height * vertical_exaggeration / 2.0

    n_points = len(points)

    # Calculate perpendicular offsets for ribbon-style mesh
    # For each point, compute the perpendicular direction to the track
    half_width = ribbon_width_mm / 2

    # Calculate track direction vectors
    dx = np.zeros(n_points)
    dy = np.zeros(n_points)

    # Use central differences for interior points, forward/backward for endpoints
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

    # Normalize direction vectors
    lengths = np.sqrt(dx ** 2 + dy ** 2)
    lengths[lengths == 0] = 1  # Avoid division by zero
    dx = dx / lengths
    dy = dy / lengths

    # Perpendicular vectors (rotate 90 degrees)
    perp_x = -dy
    perp_y = dx

    # Calculate inner and outer edge coordinates
    inner_x = x_coords - perp_x * half_width
    inner_y = y_coords - perp_y * half_width
    outer_x = x_coords + perp_x * half_width
    outer_y = y_coords + perp_y * half_width

    # Build vertices
    vertices = []
    faces = []

    # Vertex layout (4 groups of n_points each):
    # 0 to n-1: inner bottom
    # n to 2n-1: inner top
    # 2n to 3n-1: outer bottom
    # 3n to 4n-1: outer top

    # Inner edge vertices - ribbon sits on top of base plate
    inner_bottom = [(inner_x[i], inner_y[i], base_height_mm) for i in range(n_points)]
    inner_top = [(inner_x[i], inner_y[i], z_coords[i]) for i in range(n_points)]

    # Outer edge vertices
    outer_bottom = [(outer_x[i], outer_y[i], base_height_mm) for i in range(n_points)]
    outer_top = [(outer_x[i], outer_y[i], z_coords[i]) for i in range(n_points)]

    vertices.extend(inner_bottom)
    vertices.extend(inner_top)
    vertices.extend(outer_bottom)
    vertices.extend(outer_top)

    # Inner wall faces (looking from inside the ribbon)
    for i in range(n_points - 1):
        faces.append([i, i + n_points, i + 1])
        faces.append([i + 1, i + n_points, i + n_points + 1])

    # Outer wall faces (looking from outside the ribbon)
    offset = 2 * n_points
    for i in range(n_points - 1):
        faces.append([offset + i, offset + i + 1, offset + i + n_points])
        faces.append([offset + i + 1, offset + i + n_points + 1, offset + i + n_points])

    # Top face (profile surface)
    for i in range(n_points - 1):
        inner_top_idx = n_points + i
        outer_top_idx = 3 * n_points + i
        faces.append([inner_top_idx, inner_top_idx + 1, outer_top_idx])
        faces.append([inner_top_idx + 1, outer_top_idx + 1, outer_top_idx])

    # Bottom face of ribbon (sits on base plate)
    for i in range(n_points - 1):
        inner_bottom_idx = i
        outer_bottom_idx = 2 * n_points + i
        faces.append([inner_bottom_idx, outer_bottom_idx, inner_bottom_idx + 1])
        faces.append([inner_bottom_idx + 1, outer_bottom_idx, outer_bottom_idx + 1])

    # Start cap (first point)
    faces.append([0, 2 * n_points, n_points])  # inner-bottom, outer-bottom, inner-top
    faces.append([n_points, 2 * n_points, 3 * n_points])  # inner-top, outer-bottom, outer-top

    # End cap (last point)
    last = n_points - 1
    faces.append([last, n_points + last, 2 * n_points + last])
    faces.append([n_points + last, 3 * n_points + last, 2 * n_points + last])

    # Add base plate vertices
    # Base plate is width_mm x depth_mm x base_height_mm with optional bevel
    base_start_idx = len(vertices)

    # Bevel inset (45-degree bevel means inset = bevel_height)
    bevel = min(bevel_height_mm, base_height_mm, width_mm / 4, depth_mm / 4)

    # Bottom corners (z=0) - full size
    vertices.append((0, 0, 0))                      # 0: front-left-bottom
    vertices.append((width_mm, 0, 0))               # 1: front-right-bottom
    vertices.append((width_mm, depth_mm, 0))        # 2: back-right-bottom
    vertices.append((0, depth_mm, 0))               # 3: back-left-bottom

    # Top corners (z=base_height_mm) - inset by bevel amount
    vertices.append((bevel, bevel, base_height_mm))                         # 4: front-left-top
    vertices.append((width_mm - bevel, bevel, base_height_mm))              # 5: front-right-top
    vertices.append((width_mm - bevel, depth_mm - bevel, base_height_mm))   # 6: back-right-top
    vertices.append((bevel, depth_mm - bevel, base_height_mm))              # 7: back-left-top

    # Base plate faces (12 triangles for 6 faces)
    b = base_start_idx  # shorthand

    # Bottom face (z=0)
    faces.append([b+0, b+2, b+1])
    faces.append([b+0, b+3, b+2])

    # Top face (z=base_height_mm)
    faces.append([b+4, b+5, b+6])
    faces.append([b+4, b+6, b+7])

    # Front face (y=0 to y=bevel, angled)
    faces.append([b+0, b+1, b+5])
    faces.append([b+0, b+5, b+4])

    # Back face (y=depth_mm to y=depth_mm-bevel, angled)
    faces.append([b+2, b+3, b+7])
    faces.append([b+2, b+7, b+6])

    # Left face (x=0 to x=bevel, angled)
    faces.append([b+0, b+4, b+7])
    faces.append([b+0, b+7, b+3])

    # Right face (x=width_mm to x=width_mm-bevel, angled)
    faces.append([b+1, b+2, b+6])
    faces.append([b+1, b+6, b+5])

    # Add embossed text on all 4 bevel sides if specified
    if bevel_text and bevel_height_mm > 0:
        text_vertices, text_faces = create_text_mesh(
            bevel_text,
            bevel_text_height_mm,
            bevel_text_depth_mm,
            bevel_height_mm,
            base_height_mm,
            width_mm,
            depth_mm
        )
        if text_vertices:
            text_base_idx = len(vertices)
            vertices.extend(text_vertices)
            # Offset face indices for text mesh
            for face in text_faces:
                faces.append([f + text_base_idx for f in face])

    # Convert to numpy arrays
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    # Create the mesh
    elevation_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            elevation_mesh.vectors[i][j] = vertices[face[j]]

    return elevation_mesh


def gpx_to_stl(
        gpx_file: str,
        output_file: str = None,
        width_mm: float = 150.0,
        depth_mm: float = 20.0,
        base_height_mm: float = 5.0,
        vertical_exaggeration: float = 2.0,
        num_points: int = 500,
        smooth_window: int = 5,
        layout: str = "straight"
) -> str:
    """
    Convert a GPX file to an STL file.

    Args:
        gpx_file: Path to input GPX file
        output_file: Path to output STL file (default: same name as GPX with .stl extension)
        width_mm: Width of the model in mm
        depth_mm: Depth of the model in mm
        base_height_mm: Height of the base in mm
        vertical_exaggeration: Factor to exaggerate elevation differences
        num_points: Number of points to resample to
        smooth_window: Window size for elevation smoothing
        layout: "straight" for linear profile, "map" for GPS track shape

    Returns:
        Path to the created STL file
    """
    if output_file is None:
        output_file = gpx_file.rsplit('.', 1)[0] + '.stl'

    print(f"Parsing GPX file: {gpx_file}")
    points = parse_gpx(gpx_file)

    if len(points) < 2:
        raise ValueError("GPX file contains fewer than 2 valid points with elevation data")

    print(f"Found {len(points)} points")
    print(f"Total distance: {points[-1].distance / 1000:.2f} km")

    elevations = [p.elevation for p in points]
    print(f"Elevation range: {min(elevations):.0f}m - {max(elevations):.0f}m")
    print(
        f"Total elevation gain: {sum(max(0, elevations[i + 1] - elevations[i]) for i in range(len(elevations) - 1)):.0f}m")

    print(f"Smoothing elevations (window={smooth_window})...")
    points = smooth_elevations(points, smooth_window)

    print(f"Resampling to {num_points} points...")
    points = resample_points(points, num_points)

    print(f"Creating 3D mesh (layout={layout})...")
    if layout.lower() == "map":
        elevation_mesh = create_map_elevation_mesh(
            points,
            width_mm=width_mm,
            depth_mm=depth_mm,
            base_height_mm=base_height_mm,
            vertical_exaggeration=vertical_exaggeration,
            ribbon_width_mm=depth_mm  # Use depth as ribbon width for map layout
        )
    else:
        elevation_mesh = create_elevation_mesh(
            points,
            width_mm=width_mm,
            depth_mm=depth_mm,
            base_height_mm=base_height_mm,
            vertical_exaggeration=vertical_exaggeration
        )

    print(f"Saving STL file: {output_file}")
    elevation_mesh.save(output_file)

    print(f"\nModel dimensions: {width_mm}mm x {depth_mm}mm")
    print("Done!")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPX files to 3D-printable STL elevation profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ride.gpx
  %(prog)s ride.gpx -o profile.stl --width 200 --exaggeration 3
  %(prog)s climb.gpx --depth 30 --base 10
        """
    )

    parser.add_argument("gpx_file", help="Input GPX file")
    parser.add_argument("-o", "--output", help="Output STL file (default: <input>.stl)")
    parser.add_argument("--width", type=float, default=150.0,
                        help="Width of the model in mm (default: 150)")
    parser.add_argument("--depth", type=float, default=20.0,
                        help="Depth of the model in mm (default: 20)")
    parser.add_argument("--base", type=float, default=5.0,
                        help="Base height in mm (default: 5)")
    parser.add_argument("--exaggeration", type=float, default=2.0,
                        help="Vertical exaggeration factor (default: 2.0)")
    parser.add_argument("--points", type=int, default=500,
                        help="Number of points to resample to (default: 500)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Smoothing window size (default: 5)")
    parser.add_argument("--layout", choices=["straight", "map"], default="straight",
                        help="Layout style: 'straight' for linear profile, 'map' for GPS track shape (default: straight)")

    args = parser.parse_args()

    try:
        gpx_to_stl(
            args.gpx_file,
            output_file=args.output,
            width_mm=args.width,
            depth_mm=args.depth,
            base_height_mm=args.base,
            vertical_exaggeration=args.exaggeration,
            num_points=args.points,
            smooth_window=args.smooth,
            layout=args.layout
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())