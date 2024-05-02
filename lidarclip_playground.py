import argparse
import os

from mmcv.runner import load_checkpoint
from tqdm import tqdm

import torch

import clip

from lidarclip.anno_loader import build_anno_loader
from lidarclip.loader import build_loader as build_dataonly_loader
from lidarclip.model.sst import LidarEncoderSST

from train import LidarClip
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipdb;


DEFAULT_DATA_PATHS = {
    "once": "/proj/nlp4adas/datasets/once",
    "nuscenes": "/proj/berzelius-2021-92/data/nuscenes",
}

def save_image_and_pc(images, point_clouds, file_prefix=""):
    means = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cpu")
    stds = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cpu")
    
    fig = plt.figure(figsize=(12, 8))  
    for i in range(images.shape[0]):
        img = images[i]
        pc = point_clouds[i]
        print(f"Points in {i}: {pc.shape[0]}")

        ax = fig.add_axes([0, 0, 0.5, 1])
        ax.imshow((img.permute(1, 2, 0) * stds + means).numpy())
        ax.axis('off')
        
        ax = fig.add_axes([0.5, 0, 0.5, 1])
        pc = pc[pc[:,0] < 20]
        pc = pc[pc[:,1] < 10]
        pc = pc[pc[:,1] > -10]
        col = pc[:,3]
        ax.scatter(-pc[:,1], pc[:,0], s=0.1, c=col**0.3, cmap="coolwarm")
        ax.axis("scaled")
        ax.axis("off")
        ax.set_ylim(0, 20)
        ax.set_xlim(-10, 10)

        plt.savefig(f'results/{file_prefix}_image_{i}.png', dpi=300)  
        fig.clear()

    plt.close(fig)

def save_images(images):
    means = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cpu")
    stds = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cpu")
    for i in range(images.shape[0]):
        image_tensor = images[i]  
        image_array = (image_tensor.permute(1,2,0)*stds + means).numpy()
        image_array = (image_array * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_array)
        image_pil.save(f'image_{i + 1}.png')

def compute_cosine_similarity(image_features, lidar_features, clip_model):
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    lidar_features = lidar_features / lidar_features.norm(dim=1, keepdim=True)
    logit_scale = clip_model.logit_scale.exp().float()
    logits_per_image = logit_scale * image_features.float() @ lidar_features.t().float()
    return logits_per_image

def load_model(args):
    clip_model, clip_preprocess = clip.load(args.clip_version)
    print(clip_preprocess)
    lidar_encoder = LidarEncoderSST(
        "lidarclip/model/sst_encoder_only_config.py", clip_model.visual.output_dim
    )
    model = LidarClip(lidar_encoder, clip_model, 1, 1)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to("cuda")
    return model, clip_preprocess


def main(args):
    assert torch.cuda.is_available()
    model, clip_preprocess = load_model(args)
    build_loader = build_anno_loader if args.use_anno_loader else build_dataonly_loader
    loader = build_loader(
        args.data_path,
        clip_preprocess,
        batch_size=args.batch_size,
        num_workers=8,
        split=args.split,
        shuffle=True,
        enable_crop=args.enable_crop,
        blackout_crop=False,
        min_image_det_area=20000, 
        min_3d_det_points=100,
        dataset_name=args.dataset_name,
    )

    torch.manual_seed(2809)
    with torch.no_grad():
        for batch in tqdm(loader):
            images, point_clouds = batch[:2]
            # print(images.shape, point_clouds.shape)
            # save_images(images)
            save_image_and_pc(images, point_clouds, file_prefix=f"matching_viz_")
            point_clouds = [pc.to("cuda") for pc in point_clouds]
            images = [img.to("cuda") for img in images]
            images = torch.cat([i.unsqueeze(0) for i in images])
            image_features = model.clip.encode_image(images)
            lidar_features, _ = model.lidar_encoder(point_clouds)

            # similarities = torch.nn.functional.cosine_similarity(image_features, lidar_features, dim=2)
            # print(similarities)
            similarities = torch.nn.functional.cosine_similarity(image_features.unsqueeze(1), lidar_features.unsqueeze(0), dim=2)
            print(similarities)
            print(compute_cosine_similarity(image_features, lidar_features, model.clip))
            # assert False
            ipdb.set_trace()
            

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="/home/ubuntu/checkpoints/contrasive-latest/epoch=9-step=1000.ckpt", help="Full path to the checkpoint file"
    )
    # parser.add_argument("--clip-version", type=str, default="ViT-L/14")
    parser.add_argument("--clip-version", type=str, default="ViT-B/32")
    parser.add_argument("--data-path", type=str, default="/home/ubuntu/data_tars/")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--prefix", type=str, default="/features/cached")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-anno-loader", action="store_true")
    parser.add_argument("--dataset-name", type=str, default="once", choices=["once", "nuscenes"])
    parser.add_argument("--enable-crop", type=bool, default=True)
    parser.add_argument("--blackout-crop", type=bool, default=False)
    parser.add_argument("--min-image-det-area", type=float, default=20000)
    parser.add_argument("--min-3d-det-points", type=float, default=100)
    args = parser.parse_args()
    if not args.data_path:
        args.data_path = DEFAULT_DATA_PATHS[args.dataset_name]
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
