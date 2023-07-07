训练: python3.8 train.py --config ./motor/config_v5s_conv.yaml

测试: python3.8 test.py --config ./motor/config_v5s_conv.yaml

推理: python3.8 detect.py --weights /path/weights.pt --source /path/imgs_dir --save_dir /path/save_dir --img_size 736 416 --conf_thres 0.2 --iou_thres 0.3
