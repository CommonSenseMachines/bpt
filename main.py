import yaml
import torch
import os
import argparse
import trimesh
import numpy as np
from model.serializaiton import BPT_deserialize
from model.model import MeshTransformer
from utils import joint_filter, Dataset
from model.data_utils import to_mesh
from pathlib import Path
from utils_folder.progress_tracker_client import checkpoint
import logging
# prepare arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/BPT-pc-open-8k-8-16.yaml')
parser.add_argument('--model_path', type=str)
parser.add_argument('--input_dir', default=None, type=str)
parser.add_argument('--input_path', default=None, type=str)
parser.add_argument('--out_dir', default="output", type=str)
parser.add_argument('--input_type', choices=['mesh','pc_normal'], default='mesh')
parser.add_argument('--output_path', type=str, default='output')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_variations', type=int, default=1)
parser.add_argument('--run_parts', type=bool, default=False)
parser.add_argument('--temperature', type=float, default=0.5)  # key sampling parameter
parser.add_argument('--condition', type=str, default='pc')
args = parser.parse_args()


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # prepare model with fp16 precision
    model = MeshTransformer(
        dim = config['dim'],
        attn_depth = config['depth'],
        max_seq_len = config['max_seq_len'],
        dropout = config['dropout'],
        mode = config['mode'],
        num_discrete_coors= 2**int(config['quant_bit']),
        block_size = config['block_size'],
        offset_size = config['offset_size'],
        conditioned_on_pc = config['conditioned_on_pc'],
        use_special_block = config['use_special_block'],
        encoder_name = config['encoder_name'],
        encoder_freeze = config['encoder_freeze'],
    )
    model.load(args.model_path)
    model = model.eval()
    model = model.half()
    model = model.cuda()
    num_params = sum([param.nelement() for param in model.decoder.parameters()])
    print('Number of parameters: %.2f M' % (num_params / 1e6))
    print(f'Block Size: {model.block_size} | Offset Size: {model.offset_size}')

    # prepare data
    if args.input_dir is not None:
        if os.path.isfile(args.input_dir):
            input_list = [Path(args.input_dir).name]
            args.input_dir = Path(args.input_dir).parent
        else:
            input_list = os.listdir(args.input_dir)
        input_list = sorted(input_list)
        if args.input_type == 'pc_normal':
            # npy file with shape (n, 6):
            # point_cloud (n, 3) + normal (n, 3)
            input_list = [os.path.join(args.input_dir, x) for x in input_list if x.endswith('.npy')]
        else:
            # mesh file (e.g., obj, ply, glb)
            input_list = [os.path.join(args.input_dir, x) for x in input_list if not os.path.exists(os.path.join(args.output_path, Path(x).stem + f"_mesh_{args.batch_size - 1}.obj"))]
        dataset = Dataset(args.input_type, input_list, args.run_parts)

    elif args.input_path is not None:
        dataset = Dataset(args.input_type, [args.input_path], args.run_parts)

    else:
        raise ValueError("input_dir or input_path must be provided.")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last = False,
        shuffle = False,
    )

    os.makedirs(args.output_path, exist_ok=True)
    with torch.no_grad():
        for it, data in enumerate(dataloader):
            if args.condition == 'pc':
                # generate codes with model
                codes = model.generate(
                    batch_size = args.batch_size * args.num_variations,
                    temperature = args.temperature,
                    pc = data['pc_normal'].cuda().half().repeat(args.num_variations,1,1),
                    filter_logits_fn = joint_filter,
                    filter_kwargs = dict(k=50, p=0.95),
                    return_codes=True,
                )
            coords = []
            try:
                # decoding codes to coordinates
                for i in range(len(codes)):
                    code = codes[i]
                    code = code[code != model.pad_id].cpu().numpy()
                    vertices = BPT_deserialize(
                        code, 
                        block_size = model.block_size, 
                        offset_size = model.offset_size,
                        use_special_block = model.use_special_block,
                    )
                    coords.append(vertices)
            except:
                coords.append(np.zeros(3, 3))
            checkpoint("code_decoding")

            # convert coordinates to mesh
            for var_idx in range(args.num_variations):
                for batch_idx in range(args.batch_size):
                    uid = data['uid'][batch_idx]
                    vertices = coords[args.batch_size * var_idx + batch_idx]
                    faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
                    mesh = to_mesh(vertices, faces, transpose=False, post_process=True)
                    
                    # Apply denormalization to restore original dimensions
                    if args.input_type == 'mesh':
                        mesh = dataset.denormalize_mesh(mesh, uid)
                    
                    num_faces = len(mesh.faces)
                    # set the color for mesh
                    # face_color = np.array([120, 154, 192, 255], dtype=np.uint8)
                    # face_colors = np.tile(face_color, (num_faces, 1))
                    # mesh.visual.face_colors = face_colors
                    mesh.export(f'{args.output_path}/{uid}_mesh_{var_idx}.obj')

                    # save pc
                    if args.condition == 'pc':
                        pcd = data['pc_normal'][0].cpu().numpy()
                        point_cloud = trimesh.points.PointCloud(pcd[..., 0:3])
                        point_cloud.export(f'{args.output_path}/{uid}_pc.ply', "ply")
            checkpoint("mesh_export")
    
    # Log maximum VRAM usage at the end
    if torch.cuda.is_available():
        max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
        max_memory_reserved = torch.cuda.max_memory_reserved() / (1024**3)    # Convert to GB
        current_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        current_memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # Convert to GB
        
        logger.info("="*50)
        logger.info("GPU MEMORY USAGE SUMMARY")
        logger.info("="*50)
        logger.info(f"Max Memory Allocated: {max_memory_allocated:.2f} GB")
        logger.info(f"Max Memory Reserved: {max_memory_reserved:.2f} GB")
        logger.info(f"Current Memory Allocated: {current_memory_allocated:.2f} GB")
        logger.info(f"Current Memory Reserved: {current_memory_reserved:.2f} GB")
        logger.info("="*50)
        
        print(f"\n🚀 INFERENCE COMPLETE! Max VRAM Usage: {max_memory_allocated:.2f} GB (Allocated) / {max_memory_reserved:.2f} GB (Reserved)")
    else:
        logger.info("CUDA not available - no GPU memory tracking performed")
