"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time
import os
pj = os.path.join

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load

from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        # default='/mnt/ActionRecog/weights/yolo_weights/UAV_VisDrone_bdd100k/weights/best.pt',
                        default='/mnt/projects/ObjectDetection/weights/yolo_weights/Fall_v5s_conv/weights/best.pt',
                        help='weights path')  # from yolov5/models/
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--framework',default='yolov5sc-people')
    parser.add_argument('--date',default='2022xxxx')
    parser.add_argument('--img-size', nargs='+', type=int, default=[736, 416], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--fast',action='store_true')
    parser.add_argument('--img',type=str,default=None)
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)

    # model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    # labels = model.names
    model =

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    print([i.shape for i in y])
    # TorchScript export
    if opt.compact:
        """
        必须pytorch版本一致时得出的pt才能用
        """
        # try:
        #     print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        #     f = opt.weights.replace('.pt', '-1.6.0.pt')  # filename
        #     ts = torch.jit.trace(model, img)
        #     ts.save(f)
        #     print('TorchScript export success, saved as %s' % f)
        # except Exception as e:
        #     print('TorchScript export failure: %s' % e)
    else:
        try:
            from rknn.api import RKNN
            rknn = RKNN()
            print('--> Config model')
            rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                        reorder_channel='0 1 2',
                        target_platform='rk3399pro',
                        optimization_level=3,
                        quantize_input_node=QUANTIZE_ON,
                        output_optimize=1,
                        )
            print('done')

            # Load Pytorch model
            print('--> Loading model')
            dtype = 'int8'
            w, h = opt.img_size
            pre_compile = True if opt.fast else False
            pre_compile_str = 'fast' if pre_compile else 'slow'
            model_pt_full_name = opt.weights
            model_rknn_full_name = pj('rknn/', '{}_{}_{:d}x{:d}_{}.pt'.format(opt.framework,opt.date,
                                                                                   w,h, pre_compile_str))


            input_size_list = [[3, h, w]]
            ret = rknn.load_pytorch(model=model_pt_full_name, input_size_list=input_size_list)
            if ret != 0:
                print('Load Pytorch model failed!')
                exit(ret)
            print('done')
            print('--> Building model')

            ret = rknn.build(do_quantization=True, dataset='./datasets.txt', pre_compile=pre_compile)
            if ret != 0:
                print('Build model failed!')
                exit(ret)
            print('done')

            # Export RKNN model
            print('--> Export RKNN model')
            ret = rknn.export_rknn(model_rknn_full_name)
            if ret != 0:
                print('Export {} failed!'.format(model_rknn_full_name))
                exit(ret)
            print('done')

            # fast模式下， cpu上不能进行模拟推理
            if not opt.fast:
                print('--> Import RKNN model and infering')
                ret = rknn.load_rknn(model_rknn_full_name)
                # Set inputs
                # img = cv2.imread('./space_shuttle_224.jpg')
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                test_img_path = args.img
                if test_img_path is None or test_img_path == '':
                    test_img_path = 'test.jpg'
                    print('using default image for test = {}'.format(test_img_path))

                else:
                    print('using image for test = {}'.format(test_img_path))

                import cv2
                img = cv2.imread(test_img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, dsize=(w, h))

                # Init runtime environment
                print('--> Init runtime environment')
                ret = rknn.init_runtime()
                if ret != 0:
                    print('Init runtime environment failed')
                    exit(ret)
                print('done')

                # Inference
                print('--> Running model')
                outputs = rknn.inference(inputs=[img])
                print(outputs[0][0].shape)
                print(outputs[0][0])  # if model already softmax
                print('done')

        except Exception as e:
            print('ONNX export failure: %s' % e)

    # CoreML export
    """
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
    """
