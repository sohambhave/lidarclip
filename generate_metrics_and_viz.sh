# python image_lidar_detection_matching.py --exp_name cropping_min_area_20000 --enable-crop
# python image_lidar_detection_matching.py --exp_name cropping --enable-crop
# python image_lidar_detection_matching.py --exp_name blackout --enable-crop --blackout_crop
# python image_lidar_detection_matching.py --exp_name cropping_min_area_20000_min_3d_det_points_200 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200
# python image_lidar_detection_matching.py --exp_name cropping_min_area_20000 --enable-crop --min_image_det_area 20000

# python image_lidar_detection_matching.py --exp_name cropping_min_area_20000_min_3d_det_points_200_new_ckpt10 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/acdym3ts/last-98.ckpt

#python image_lidar_detection_matching.py --exp_name sim_e95 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/acdym3ts/epoch=95-step=39750.ckpt
#python image_lidar_detection_matching.py --exp_name sim_e98 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/acdym3ts/epoch=98-step=41000.ckpt
#python image_lidar_detection_matching.py --exp_name sim_e101 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/acdym3ts/epoch=101-step=42250.ckpt

#python image_lidar_detection_matching.py --exp_name con_e99 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/h470bgok/epoch=99-step=10500.ckpt
#python image_lidar_detection_matching.py --exp_name con_e114 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/h470bgok/epoch=114-step=12000.ckpt
# Baseline first
python image_lidar_detection_matching.py --exp_name baseline_lidarclip --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --visualize-similarity-matrix
python image_lidar_detection_matching.py --exp_name sim_best_e2 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/similarity-latest/epoch=2-step=250.ckpt --visualize-similarity-matrix
python image_lidar_detection_matching.py --exp_name sim_best_e4 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/similarity-latest/epoch=4-step=500.ckpt --visualize-similarity-matrix
python image_lidar_detection_matching.py --exp_name sim_best_e9 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/similarity-latest/epoch=9-step=1000.ckpt --visualize-similarity-matrix
python image_lidar_detection_matching.py --exp_name sim_best_last --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/similarity-latest/last.ckpt --visualize-similarity-matrix

python image_lidar_detection_matching.py --exp_name con_best_e7 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/contrasive-latest/epoch=7-step=750.ckpt --visualize-similarity-matrix
python image_lidar_detection_matching.py --exp_name con_best_e9 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/contrasive-latest/epoch=9-step=1000.ckpt --visualize-similarity-matrix
python image_lidar_detection_matching.py --exp_name con_best_e16 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/contrasive-latest/epoch=16-step=1750.ckpt --visualize-similarity-matrix
python image_lidar_detection_matching.py --exp_name con_best_last --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/contrasive-latest/last.ckpt --visualize-similarity-matrix
python image_lidar_detection_matching.py --exp_name sim_best_e2 --enable-crop --min_image_det_area 20000 --min_3d_det_points 200 --checkpoint /home/ubuntu/checkpoints/similarity-latest/epoch=2-step=250.ckpt --visualize-similarity-matrix