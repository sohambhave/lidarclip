# python train_crops.py --data-dir=/home/ubuntu/data_tars/ --checkpoint-save-dir=/home/ubuntu/checkpoints/ --name=crops_b_32 --checkpoint /home/ubuntu/checkpoints/4q7gpsje/last.ckpt --clip-model ViT-B/32

python train_crops.py --data-dir=/home/ubuntu/data_tars/ --checkpoint-save-dir=/home/ubuntu/checkpoints/ --name=crops_b_32_contrastive --checkpoint /home/ubuntu/checkpoints/vit_b_32.ckpt --clip-model ViT-B/32 --load-only-model  --loss-function contrastive --batch-size 128
python train_crops.py --data-dir=/home/ubuntu/data_tars/ --checkpoint-save-dir=/home/ubuntu/checkpoints/ --name=crops_b_32_similarity_mse --checkpoint /home/ubuntu/checkpoints/vit_b_32.ckpt --clip-model ViT-B/32 --load-only-model --batch-size 128 
# python train_crops.py --data-dir=/home/ubuntu/data_tars/ --checkpoint-save-dir=/home/ubuntu/checkpoints/ --name=crops_b_32 --checkpoint /home/ubuntu/checkpoints/acdym3ts/last.ckpt --clip-model ViT-B/32 --resume-wandb-logging
# python train_crops.py --data-dir=/home/ubuntu/data_tars/ --checkpoint-save-dir=/home/ubuntu/checkpoints/ --name=crops_l_14 --checkpoint /home/ubuntu/checkpoints/vit_l_14.ckpt

# python train.py --data-dir=/home/ubuntu/data_tars/ --checkpoint-save-dir=/home/ubuntu/checkpoints/ --name=b_32 --checkpoint /home/ubuntu/checkpoints/vit_b_32.ckpt --clip-model ViT-B/32
