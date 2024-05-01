import argparse
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
from tqdm import tqdm

import clip

from mmcv.runner import load_checkpoint
from lidarclip.anno_loader import build_anno_loader
from lidarclip.loader import build_loader as build_dataonly_loader
from lidarclip.model.sst import LidarEncoderSST
from train import LidarClip

metrics = {}

def load_model(args):
    # Load the CLIP model for images and the specialized Lidar encoder for lidar data
    clip_model, clip_preprocess = clip.load(args.clip_version, device='cuda')
    lidar_encoder = LidarEncoderSST(
        "lidarclip/model/sst_encoder_only_config.py",
        clip_model.visual.output_dim
    )
    model = LidarClip(lidar_encoder=lidar_encoder, clip_model=clip_model, batch_size=args.batch_size, epoch_size=1)
    load_checkpoint(model, args.checkpoint, map_location='cuda')
    model.to('cuda')
    return model, clip_preprocess


def zero_shot_matching(args):
    model, clip_preprocess = load_model(args)
    build_loader = build_anno_loader if args.use_anno_loader else build_dataonly_loader
    loader = build_loader(
        args.data_path,
        clip_preprocess,
        batch_size=args.batch_size,
        num_workers=8,
        split=args.split,
        shuffle=True,
        blackout_crop=args.blackout_crop,
        enable_crop=args.enable_crop,
        min_image_det_area=args.min_image_det_area,
        min_3d_det_points=args.min_3d_det_points,
    )

    torch.manual_seed(args.manual_seed)
    
    # Initialize accumulators for metrics
    image_query_accuracy_sum = 0
    lidar_query_accuracy_sum = 0
    image_query_top_k_accuracy_sum = 0
    lidar_query_top_k_accuracy_sum = 0
    num_batches_processed = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Matching Image and Lidar Crops"):
            images, point_clouds = batch[:2]
            # print(f"{len(images)=}")
            # print(f"{len(point_clouds)=}")

            # Convert images and point clouds to CUDA
            images = [img.to("cuda") for img in images]
            point_clouds = [pc.to("cuda") for pc in point_clouds]

            # Batch images for processing
            images = torch.cat([img.unsqueeze(0) for img in images])

            # Generate embeddings from images and point clouds
            image_features = model.clip.encode_image(images)
            lidar_features, _ = model.lidar_encoder(point_clouds)
            # print(f"Shape of image_features: {image_features.shape}")
            # print(f"Shape of lidar_features: {lidar_features.shape}")
            image_features = image_features.unsqueeze(1)
            lidar_features = lidar_features.unsqueeze(0)
            # print(f"after unsqueeze(1): Shape of image_features: {image_features.shape}")
            # print(f"after unsqueeze(0): Shape of lidar_features: {lidar_features.shape}")

            # Compute cosine similarities between all pairs of image and lidar features
            # Cosine similarity expects features in (batch_size, features) format
            similarities = torch.nn.functional.cosine_similarity(image_features, lidar_features, dim=2)

             # Update accumulators for each metric
            if args.compute_matching_accuracy:
                image_query_accuracy = compute_accuracy(similarities, 'image_query')
                lidar_query_accuracy = compute_accuracy(similarities, 'lidar_query')
                image_query_accuracy_sum += image_query_accuracy
                lidar_query_accuracy_sum += lidar_query_accuracy

            if args.compute_top_k_matching_accuracy:
                k = args.compute_top_k_matching_accuracy
                image_query_top_k_accuracy = compute_top_k_accuracy(similarities, k=k, method='image_query')
                lidar_query_top_k_accuracy = compute_top_k_accuracy(similarities, k=k, method='lidar_query')
                image_query_top_k_accuracy_sum += image_query_top_k_accuracy
                lidar_query_top_k_accuracy_sum += lidar_query_top_k_accuracy

            if args.visualize_similarity_matrix:
                # Visualization of similarity matrix (implement this function as needed)
                visualize_similarity_matrix(similarities.cpu().numpy(),
                                            filename=f"results/{args.exp_name}_similarity_matrix.png")
            
            num_batches_processed += 1

            if args.num_batches and num_batches_processed >= args.num_batches:
                break

    # Calculate average accuracies and update the metrics dictionary
    if args.compute_matching_accuracy:
        metrics["Image Query Accuracy"] = image_query_accuracy_sum / num_batches_processed
        metrics["Lidar Query Accuracy"] = lidar_query_accuracy_sum / num_batches_processed

    if args.compute_top_k_matching_accuracy:
        metrics[f"Image Query Top-{k} Accuracy"] = image_query_top_k_accuracy_sum / num_batches_processed
        metrics[f"Lidar Query Top-{k} Accuracy"] = lidar_query_top_k_accuracy_sum / num_batches_processed


def visualize_similarity_matrix(similarities, filename="results/similarity_matrix.png"):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(similarities, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Cosine Similarity between Image and Lidar Scans')
    plt.xlabel('Lidar Scan Index')
    plt.ylabel('Image Index')
    # Reverse the y-axis to have the image index increase from bottom to top
    # ax.set_ylim(len(similarities), 0)
    ax.set_ylim(0, len(similarities))
    # plt.show()
    # Save the figure
    print("Saving viz to {filename}...")
    plt.savefig(filename)  # Saves the heatmap to a file
    plt.close()  # Close the figure to free up memory

def compute_accuracy(similarities, method='image_query'):
    """
    Compute matching accuracy using the cosine similarity matrix.

    Args:
    similarities (Tensor): The cosine similarity matrix of shape (num_images, num_lidars).
    method (str): Method to compute accuracy, either 'image_query' or 'lidar_query'.
    """
    # Assume diagonal indices are the correct matches
    if method == 'image_query':
        # Find the index of the max similarity for each image across all lidar embeddings
        predicted_indices = torch.argmax(similarities, dim=1)
        correct_matches = torch.eq(predicted_indices, torch.arange(len(predicted_indices)).to(similarities.device))
    elif method == 'lidar_query':
        # Find the index of the max similarity for each lidar across all image embeddings
        predicted_indices = torch.argmax(similarities, dim=0)
        correct_matches = torch.eq(predicted_indices, torch.arange(len(predicted_indices)).to(similarities.device))
    else:
        raise ValueError("Unknown method specified for compute_accuracy.")

    accuracy = torch.mean(correct_matches.float()) * 100  # Convert to percentage
    return accuracy.item()

def compute_top_k_accuracy(similarities, k=1, method='image_query'):
    """
    Compute matching accuracy considering top-k matches using the cosine similarity matrix.
    """
    if method == 'image_query':
        # Get the top-k indices along the lidar dimension for each image
        top_k_indices = torch.topk(similarities, k=k, dim=1).indices
        correct_indices = torch.arange(len(top_k_indices)).to(similarities.device).unsqueeze(1)
        correct_matches = (top_k_indices == correct_indices).any(dim=1)
    elif method == 'lidar_query':
        # Get the top-k indices along the image dimension for each lidar
        top_k_indices = torch.topk(similarities, k=k, dim=0).indices
        correct_indices = torch.arange(len(top_k_indices)).to(similarities.device).unsqueeze(1)
        correct_matches = (top_k_indices == correct_indices).any(dim=1)
    else:
        raise ValueError("Unknown method specified for compute_top_k_accuracy.")

    accuracy = torch.mean(correct_matches.float()) * 100  # Convert to percentage
    return accuracy.item()


def write_metrics_to_file(args):
    if bool(len(metrics)):
        filename = f"results/{args.exp_name}_metrics.csv"
        print(f"Writing the following metrics to file: {metrics}")
        with open(filename, 'w') as f:
            # Write headers based on the keys
            metric_names = list(metrics.keys())
            headers = ["Experiment Name"] + metric_names
            f.write(','.join(headers) + '\n')
            # Write values in the same order as headers
            values = [args.exp_name] + [str(metrics[key]) for key in metric_names]
            f.write(','.join(values) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot matching using LidarClip.")
    parser.add_argument(
        "--checkpoint", type=str, default="/home/ubuntu/checkpoints/vit_b_32.ckpt", help="Full path to the checkpoint file"
    )
    parser.add_argument("--manual-seed", type=int, default=2809, help="Change seed to select different batch.")
    parser.add_argument("--clip-version", type=str, default="ViT-B/32")
    parser.add_argument("--data-path", type=str, default="/home/ubuntu/data_tars/")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of samples per batch.")
    parser.add_argument("--use-anno-loader", action='store_true', help="Use the loader that accounts for annotations.")
    parser.add_argument("--enable-crop", action='store_true', help="crop lidar and image frustums / patches before feature extraction.")
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--min_image_det_area", type=int, default=0, help="")
    parser.add_argument("--min_3d_det_points", type=int, default=0, help="")
    parser.add_argument("--blackout_crop", action='store_true', help="")
    parser.add_argument("--num_batches", type=int, default=1, help="0 indicates all batches")

    # Visualization
    parser.add_argument("--visualize-similarity-matrix", action="store_true", help="Generate heatmap.")

    # Metrics
    parser.add_argument("--compute_matching_accuracy", default=True, help="Compute the accuracy of matches.")
    parser.add_argument("--compute_top_k_matching_accuracy", type=int, default=5, help="Compute the top-k matching accuracy where k is the number of top ranks to consider for a match.")

    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # TODO(sid): Remove these over-rides
    args.visualize_similarity_matrix = False
    args.num_batches = 0

    if args.visualize_similarity_matrix:
        assert args.num_batches == 1, "Can only visualize similarity_matrix for 1 batch at a time"

    zero_shot_matching(args)
    write_metrics_to_file(args)
