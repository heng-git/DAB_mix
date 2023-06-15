#nohup python -m torch.distributed.launch --nproc_per_node=2 \
#  main.py -m dab_detr \
#  --output_dir /logs/DABDETR/R50  \
#  --batch_size 8 \
#  --epochs 12 \
#  --lr_drop 10 \
#  --coco_path /Object_detection/gly/datasets/coco\
#>std.log 2>&1 &

nohup python -m torch.distributed.launch --nproc_per_node=4 \
  main.py -m dab_detr \
  --output_dir logs/DABDETR/R50  \
  --batch_size 4 \
  --epochs 12 \
  --lr_drop 10 \
 --coco_path /var/lib/docker/user1/ZSH/datasets/coco \
>std.log 2>&1 &