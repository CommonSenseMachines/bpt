import trimesh
import numpy as np
import subprocess
import point_cloud_utils as pcu
from x_transformers.autoregressive_wrapper import top_p, top_k
from calculate_stats import calculate_stats
import pymeshlab as pml
import tempfile

# --- Filter thresholds (adjust as needed) ---
MIN_VOL_RATIO = 0.01      # Minimum 1% of total volume
MIN_AREA_RATIO = 0.01     # Minimum 1% of total surface area
MIN_FACE_COUNT = 1000     # Or at least 1000 faces

class Dataset:
    '''
    A toy dataset for inference
    '''
    def __init__(self, input_type, input_list, run_parts):
        super().__init__()
        self.data = []
        self.normalization_params = {}  # Store normalization parameters for each uid
        self.original_meshes = {}  # Store original mesh data for quality comparison
        self.decimate_meshes = []
        self.static_parts_map = []
        
        if input_type == 'pc_normal':
            for input_path in input_list:
                # load npy
                cur_data = np.load(input_path)
                # sample 4096
                assert cur_data.shape[0] >= 4096, "input pc_normal should have at least 4096 points"
                idx = np.random.choice(cur_data.shape[0], 4096, replace=False)
                cur_data = cur_data[idx]
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})

        elif input_type == 'mesh':
            for input_path in input_list:
                part_stats, total_volume, total_area, total_faces = calculate_stats(input_path)
                # sample point cloud and normal from mesh
                if run_parts:
                    cur_data = trimesh.load(input_path)
                else:
                    cur_data = trimesh.load(input_path, force='mesh')
                
                if hasattr(cur_data, 'geometry'):  # It's a Scene with multiple geometries
                    # Calculate global normalization parameters for the entire scene
                    # global_norm_params = calculate_global_normalization_params(cur_data)
                    
                    # # Apply global normalization to the entire scene first
                    # breakpoint()
                    # apply_global_normalization_to_scene(cur_data, global_norm_params)
                    
                    for geom_name, geom in cur_data.geometry.items():
                        if isinstance(geom, trimesh.Trimesh):  # Make sure it's a mesh
                            # Store original mesh before normalization
                            
                            geom, norm_params = apply_normalize(geom, return_params=True)
                            original_mesh = geom.copy()
                            main_uid = input_path.split('/')[-1].split('.')[0]
                            uid = f"{input_path.split('/')[-1].split('.')[0]}_{geom_name}"
                            self.normalization_params[uid] = norm_params
                            
                            # Store original mesh data for quality comparison
                            self.original_meshes[uid] = original_mesh

                            v = part_stats[geom_name]["volume"]
                            a = part_stats[geom_name]["area"]
                            f = part_stats[geom_name]["faces"]
                            a_ratio = a / total_area
                            process = (a_ratio > MIN_AREA_RATIO) or (f >= MIN_FACE_COUNT)
                            if not process:
                                geom = self.denormalize_mesh(geom, uid)
                                self.static_parts_map.append({'geo': geom, 'uid': uid, 'main_uid': main_uid})
                            elif v < 0.1:
                                geom = self.denormalize_mesh(geom, uid)
                                print(f"decimate mesh: {uid}")
                                self.decimate_meshes.append({'geo': geom, 'uid': uid, 'main_uid': main_uid})
                            else:
                                pc_data = sample_pc_from_mesh(geom, pc_num=4096, with_normal=True)
                                self.data.append({'pc_normal': pc_data, 'uid': uid, 'main_uid': main_uid})
                else:  # It's a single mesh
                    # Store original mesh before normalization
                    original_mesh = cur_data.copy()
                    
                    cur_data, norm_params = apply_normalize(cur_data, return_params=True)
                    uid = input_path.split('/')[-1].split('.')[0]
                    self.normalization_params[uid] = norm_params
                    
                    # Store original mesh data for quality comparison
                    self.original_meshes[uid] = original_mesh
                    
                    pc_data = sample_pc_from_mesh(cur_data, pc_num=4096, with_normal=True)
                    self.data.append({'pc_normal': pc_data, 'uid': uid, 'main_uid': uid})
                
        print(f"dataset total data samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}
        data_dict['pc_normal'] = self.data[idx]['pc_normal']
        data_dict['uid'] = self.data[idx]['uid']
        data_dict['main_uid'] = self.data[idx]['main_uid']

        return data_dict
    
    def get_normalization_params(self, uid):
        """Get normalization parameters for a specific uid"""
        return self.normalization_params.get(uid, None)
    
    def get_original_mesh(self, uid):
        """Get original mesh for a specific uid"""
        return self.original_meshes.get(uid, None)
    
    def denormalize_mesh(self, mesh, uid):
        """Denormalize a mesh using the stored normalization parameters"""
        norm_params = self.get_normalization_params(uid)
        if norm_params is None:
            print(f"Warning: No normalization parameters found for uid {uid}")
            return mesh
        
        return apply_denormalize(mesh, norm_params)


def joint_filter(logits, k = 50, p=0.95):
    logits = top_k(logits, k = k)
    logits = top_p(logits, thres = p)
    return logits


def calculate_global_normalization_params(scene):
    '''
    Calculate normalization parameters for an entire scene containing multiple geometries
    '''
    # Get the combined bounding box of all geometries
    global_bounds = scene.bounds
    center = (global_bounds[1] + global_bounds[0]) / 2
    scale = (global_bounds[1] - global_bounds[0]).max()
    
    norm_params = {
        'center': center,
        'scale': scale,
        'normalize_scale': 1 / scale * 2 * 0.95
    }
    
    return norm_params


def apply_global_normalization_to_scene(scene, global_norm_params):
    '''
    Apply global normalization (center + scale) to entire scene
    '''
    # Apply translation to center the scene
    scene.apply_translation(-global_norm_params['center'])
    # Apply scaling to fit in [-1,1] cube
    scene.apply_scale(global_norm_params['normalize_scale'])


def apply_normalize(mesh, return_params=False):
    '''
    normalize mesh to [-1, 1]
    '''
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale = (bbox[1] - bbox[0]).max()
    normalize_scale = 1 / scale * 2 * 0.95

    mesh.apply_translation(-center)
    mesh.apply_scale(normalize_scale)

    if return_params:
        # Return the normalization parameters needed to undo the transformation
        norm_params = {
            'center': center,
            'scale': scale,
            'normalize_scale': normalize_scale
        }
        return mesh, norm_params
    
    return mesh


def apply_denormalize(mesh, norm_params):
    '''
    Undo the normalization using the stored parameters
    '''
    # Undo the scaling
    mesh.apply_scale(1 / norm_params['normalize_scale'])
    # Undo the translation
    mesh.apply_translation(norm_params['center'])
    
    return mesh


def sample_pc_from_mesh(mesh, pc_num, with_normal=False):
    """
    Sample point cloud from an already loaded and normalized mesh
    """
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points

    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    
    return pc_normal


def sample_pc(mesh_path, pc_num, with_normal=False):

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    mesh = apply_normalize(mesh)
    
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points

    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    
    return pc_normal


def compute_chamfer_distance_mesh_to_pc(mesh, point_cloud, num_samples=4096):
    """
    Compute chamfer distance between a mesh and a point cloud.
    
    Args:
        mesh: trimesh.Trimesh object
        point_cloud: numpy array of shape (N, 3) or (N, 6) where first 3 columns are points
        num_samples: number of points to sample from mesh for comparison
    
    Returns:
        chamfer_distance: float value of chamfer distance
    """
    try:
        # Sample points from the mesh
        mesh_points, _ = mesh.sample(num_samples, return_index=True)
        
        # Extract just the point coordinates from point cloud (in case it has normals)
        if point_cloud.shape[1] > 3:
            pc_points = point_cloud[:, :3]
        else:
            pc_points = point_cloud
        
        # Compute chamfer distance using point_cloud_utils
        chamfer_dist = pcu.chamfer_distance(np.array(mesh_points).astype(np.float32), pc_points.astype(np.float32))
        
        return chamfer_dist
    except Exception as e:
        print(f"Error computing chamfer distance: {e}")
        return float('inf')  # Return high value on failure

def compute_chamfer_distance_mesh_to_mesh(mesh1, mesh2, num_samples=4096):
    """
    Compute chamfer distance between a mesh and a point cloud.
    
    Args:
        mesh: trimesh.Trimesh object
        point_cloud: numpy array of shape (N, 3) or (N, 6) where first 3 columns are points
        num_samples: number of points to sample from mesh for comparison
    
    Returns:
        chamfer_distance: float value of chamfer distance
    """
    try:
        # Sample points from the mesh
        mesh_points1, _ = mesh1.sample(num_samples, return_index=True)
        mesh_points2, _ = mesh2.sample(num_samples, return_index=True)
        
        # Compute chamfer distance using point_cloud_utils
        chamfer_dist = pcu.chamfer_distance(mesh_points1, mesh_points2)
        
        return chamfer_dist
    except Exception as e:
        print(f"Error computing chamfer distance: {e}")
        return float('inf')  # Return high value on failure


def should_use_original_mesh(generated_mesh, original_mesh, threshold=0.01, num_samples=4096):
    """
    Determine if original mesh should be used instead of generated mesh
    based on chamfer distance quality metric.
    
    Args:
        generated_mesh: trimesh.Trimesh object of generated result
        original_pc: numpy array of original point cloud
        threshold: chamfer distance threshold above which to use original mesh
    
    Returns:
        bool: True if original mesh should be used, False otherwise
    """
    chamfer_dist = compute_chamfer_distance_mesh_to_mesh(generated_mesh, original_mesh, num_samples)
    return chamfer_dist > threshold

def pymeshlab2trimesh(mesh: pml.MeshSet) -> trimesh.Trimesh:
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=True) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = trimesh.load(temp_file.name)
    
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh

def trimesh2pymeshlab(mesh: trimesh.Trimesh) -> pml.MeshSet:
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=True) as temp_file:
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(temp_file.name)
        mesh = pml.MeshSet()
        mesh.load_new_mesh(temp_file.name)
    return mesh

def decimate_mesh(inp_mesh, target_fac):
    
    mesh = trimesh2pymeshlab(inp_mesh)
    
    if target_fac is not None:
        mesh.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_fac,
            qualitythr=1.0,
            preserveboundary=True,
            boundaryweight=3,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            autoclean=True,
        )
    
    return pymeshlab2trimesh(mesh)