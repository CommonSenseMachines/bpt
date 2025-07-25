#!/usr/bin/env python3
import sys
import argparse
import trimesh
import point_cloud_utils as pcu

# --- Filter thresholds (adjust as needed) ---
MIN_VOL_RATIO = 0.01      # Minimum 1% of total volume
MIN_AREA_RATIO = 0.01     # Minimum 1% of total surface area
MIN_FACE_COUNT = 1000     # Or at least 1000 faces
# --------------------------------------------

def make_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    from utils import apply_normalize
    mesh = apply_normalize(mesh, return_params=False)
    if mesh.is_watertight and mesh.volume is not None:
        return mesh
    filled = mesh.copy()
    v_watertight, f_watertight = pcu.make_mesh_watertight(filled.vertices, filled.faces, resolution=10_000)
    wt_mesh = trimesh.Trimesh(v_watertight, f_watertight)
    return wt_mesh

def calculate_stats(path: str, verbose: bool = False) -> None:
    cur_data = trimesh.load(path, force='scene')

    if hasattr(cur_data, "geometry"):  # Scene with multiple geometries
        parts = {name: geom for name, geom in cur_data.geometry.items()}
    elif isinstance(cur_data, trimesh.Trimesh):  # Single mesh
        parts = {path: cur_data}
    else:
        print("Unsupported mesh type:", type(cur_data))
        sys.exit(1)

    part_stats = {}
    total_volume = 0.0
    total_area = 0.0
    total_faces = 0

    for name, mesh in parts.items():
        wt_mesh = make_watertight(mesh)
        volume = wt_mesh.volume or 0.0
        area = mesh.area or 0.0
        n_faces = len(mesh.faces)
        total_volume += volume
        total_area += area
        total_faces += n_faces
        part_stats[name] = {
            "volume": volume,
            "area": area,
            "faces": n_faces
        }

    if total_volume == 0 or total_area == 0 or total_faces == 0:
        print("Cannot compute percentages — some totals are zero.")
        sys.exit(1)

    if verbose:
        print(f"{'Part name':<30} {'Volume':>10} {'Vol%':>6}  {'Area':>10} {'Area%':>6}  {'Faces':>8} {'Face%':>6}  {'Keep':>5}")
        print("-" * 100)

        kept_count = 0
        for name, stat in part_stats.items():
            v = stat["volume"]
            a = stat["area"]
            f = stat["faces"]
            v_ratio = v / total_volume
            a_ratio = a / total_area
            f_ratio = f / total_faces

            keep = (v_ratio > MIN_VOL_RATIO) or (a_ratio > MIN_AREA_RATIO) or (f >= MIN_FACE_COUNT)
            keep_str = "Yes" if keep else "No"
            if keep:
                kept_count += 1

            print(f"{name:<30} "
                f"{v:10.4g} {v_ratio:6.2%}  "
                f"{a:10.4g} {a_ratio:6.2%}  "
                f"{f:8d} {f_ratio:6.2%}  "
                f"{keep_str:>5}")

        print(f"\n✅ Kept {kept_count} out of {len(part_stats)} parts")

    return part_stats, total_volume, total_area, total_faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print per-part volume, area, and face count ratios")
    parser.add_argument("mesh_path", help="Path to mesh file (obj/ply/glb/…)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    calculate_stats(args.mesh_path, args.verbose)
