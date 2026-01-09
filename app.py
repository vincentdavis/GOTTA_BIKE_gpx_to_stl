"""
GPX to STL Converter - Streamlit App
Converts GPS track files to 3D-printable elevation profile models.
"""

import io
import tempfile
from pathlib import Path

import gpxpy
import numpy as np
import streamlit as st
from stl import mesh

from gps_stl import (
    TrackPoint,
    haversine_distance,
    smooth_elevations,
    resample_points,
    create_elevation_mesh,
)


def parse_gpx_from_bytes(gpx_bytes: bytes) -> list[TrackPoint]:
    """Parse GPX file from bytes and return a list of TrackPoints with cumulative distance."""
    gpx = gpxpy.parse(gpx_bytes.decode('utf-8'))

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


def mesh_to_plotly_data(stl_mesh: mesh.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert STL mesh to plotly-compatible data."""
    # Extract vertices from mesh
    vectors = stl_mesh.vectors

    # Flatten all vertices
    all_vertices = vectors.reshape(-1, 3)

    # Create unique vertices and face indices
    unique_vertices, inverse_indices = np.unique(
        all_vertices, axis=0, return_inverse=True
    )

    # Reshape indices to faces (each face has 3 vertices)
    faces = inverse_indices.reshape(-1, 3)

    x = unique_vertices[:, 0]
    y = unique_vertices[:, 1]
    z = unique_vertices[:, 2]

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    return x, y, z, i, j, k


def create_stl_preview(stl_mesh: mesh.Mesh):
    """Create a 3D preview of the STL mesh using plotly."""
    import plotly.graph_objects as go

    x, y, z, i, j, k = mesh_to_plotly_data(stl_mesh)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='lightblue',
            opacity=1.0,
            flatshading=True,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5
            ),
            lightposition=dict(x=100, y=200, z=300)
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title='Width (mm)',
            yaxis_title='Depth (mm)',
            zaxis_title='Height (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )

    return fig


def gpx_bytes_to_stl(
    gpx_bytes: bytes,
    width_mm: float = 150.0,
    depth_mm: float = 20.0,
    base_height_mm: float = 5.0,
    vertical_exaggeration: float = 2.0,
    num_points: int = 500,
    smooth_window: int = 5
) -> tuple[mesh.Mesh, dict]:
    """
    Convert GPX bytes to STL mesh.

    Returns:
        Tuple of (mesh, stats_dict)
    """
    points = parse_gpx_from_bytes(gpx_bytes)

    if len(points) < 2:
        raise ValueError("GPX file contains fewer than 2 valid points with elevation data")

    elevations = [p.elevation for p in points]
    stats = {
        'num_points': len(points),
        'total_distance_km': points[-1].distance / 1000,
        'min_elevation': min(elevations),
        'max_elevation': max(elevations),
        'elevation_gain': sum(max(0, elevations[i + 1] - elevations[i]) for i in range(len(elevations) - 1))
    }

    # Process points
    points = smooth_elevations(points, smooth_window)
    points = resample_points(points, num_points)

    # Create mesh
    stl_mesh = create_elevation_mesh(
        points,
        width_mm=width_mm,
        depth_mm=depth_mm,
        base_height_mm=base_height_mm,
        vertical_exaggeration=vertical_exaggeration
    )

    return stl_mesh, stats


def mesh_to_bytes(stl_mesh: mesh.Mesh) -> bytes:
    """Convert mesh to STL bytes for download."""
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl_mesh.save(f.name)
        f.seek(0)
        stl_path = Path(f.name)

    stl_bytes = stl_path.read_bytes()
    stl_path.unlink()  # Clean up temp file
    return stl_bytes


# Page configuration
st.set_page_config(
    page_title="GPX to STL Converter",
    page_icon="🚴",
    layout="wide"
)

st.title("🚴 GPX to STL Converter")
st.markdown("""
Convert your GPS ride files into 3D-printable elevation profiles!
Upload a GPX file from your cycling computer or app, customize the dimensions,
and download an STL file ready for 3D printing.
""")

# Sidebar for parameters
st.sidebar.header("Model Settings")

width_mm = st.sidebar.slider(
    "Width (mm)",
    min_value=50,
    max_value=300,
    value=150,
    step=10,
    help="Total width of the 3D model"
)

depth_mm = st.sidebar.slider(
    "Depth (mm)",
    min_value=10,
    max_value=50,
    value=20,
    step=5,
    help="Thickness/depth of the model"
)

base_height_mm = st.sidebar.slider(
    "Base Height (mm)",
    min_value=2,
    max_value=20,
    value=5,
    step=1,
    help="Height of the flat base"
)

vertical_exaggeration = st.sidebar.slider(
    "Vertical Exaggeration",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="Factor to exaggerate elevation differences"
)

st.sidebar.header("Processing Settings")

num_points = st.sidebar.slider(
    "Resolution (points)",
    min_value=100,
    max_value=1000,
    value=500,
    step=50,
    help="Number of points in the profile (higher = more detail)"
)

smooth_window = st.sidebar.slider(
    "Smoothing",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Smoothing window size (higher = smoother profile)"
)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload GPX File")
    uploaded_file = st.file_uploader(
        "Choose a GPX file",
        type=['gpx'],
        help="Upload a GPX file from your GPS device or cycling app"
    )

if uploaded_file is not None:
    try:
        gpx_bytes = uploaded_file.read()

        with st.spinner("Processing GPX file..."):
            stl_mesh, stats = gpx_bytes_to_stl(
                gpx_bytes,
                width_mm=float(width_mm),
                depth_mm=float(depth_mm),
                base_height_mm=float(base_height_mm),
                vertical_exaggeration=float(vertical_exaggeration),
                num_points=num_points,
                smooth_window=smooth_window
            )

        # Show stats
        with col1:
            st.success("GPX file processed successfully!")

            st.subheader("Route Statistics")
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.metric("Distance", f"{stats['total_distance_km']:.2f} km")
                st.metric("Min Elevation", f"{stats['min_elevation']:.0f} m")
            with stat_col2:
                st.metric("Elevation Gain", f"{stats['elevation_gain']:.0f} m")
                st.metric("Max Elevation", f"{stats['max_elevation']:.0f} m")

            st.subheader("Model Dimensions")
            st.write(f"**Size:** {width_mm} x {depth_mm} mm")
            st.write(f"**Points:** {stats['num_points']} original, {num_points} resampled")

        with col2:
            st.header("3D Preview")
            with st.spinner("Generating preview..."):
                fig = create_stl_preview(stl_mesh)
                st.plotly_chart(fig, use_container_width=True)

        # Download button
        st.header("Download STL")
        stl_bytes = mesh_to_bytes(stl_mesh)

        output_filename = uploaded_file.name.rsplit('.', 1)[0] + '.stl'

        st.download_button(
            label="Download STL File",
            data=stl_bytes,
            file_name=output_filename,
            mime="application/octet-stream",
            type="primary"
        )

        st.info(f"File size: {len(stl_bytes) / 1024:.1f} KB")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    with col2:
        st.header("3D Preview")
        st.info("Upload a GPX file to see the 3D preview here.")

# Footer
st.markdown("---")
st.markdown("""
**Tips for best results:**
- Use GPX files with elevation data (most cycling apps include this)
- Adjust vertical exaggeration to make flat routes more visible
- Higher resolution gives more detail but larger file sizes
- Smoothing helps reduce noise in GPS elevation data
""")