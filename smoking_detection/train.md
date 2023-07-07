nohup python train.py --config smoking_detection/config_v5s_conv.yaml >x.out 2>&1 &

python test.py --config smoking_detection/config_v5s_conv.yaml