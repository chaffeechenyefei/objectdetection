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
from yolov5_api import load_yolo_model

from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from trans_weights_to_old_pth import model_context

QUANTIZE_ON = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key',default='yolov5s-conv-9-20211104')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--fast',action='store_true')
    parser.add_argument('--img',type=str,default=None)
    opt = parser.parse_args()
    # opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    pt_weights = model_context[opt.model_key]['weights'].replace('.pt','-1.6.0.pt')
    onnx_weights = model_context[opt.model_key]['weights'].replace('.pt','.onnx')

    # yolov5_net = torch.jit.load(pt_weights, map_location=device)  # load FP32 model
    # # labels = model.names
    # # model = load_yolo_model(cfg=opt.cfg, weights=opt.weights)
    # yolov5_net = yolov5_net.eval()
    # # model_name = opt.model_key
    # # singleOutput = True
    # # pretrained = True if model_context[model_name]['weights'] is not None else False
    # # pth_weights = model_context[model_name]['weights'].replace('.pt', '.pth')
    # # yolov5_net = load_yolo_model(cfg=model_context[model_name]['cfg'], weights=pth_weights,
    # #                              singleOutput=singleOutput,
    # #                              preTrained=pretrained)
    # # yolov5_net = yolov5_net.eval().float()
    #
    # # Checks
    # # gs = int(max(model.stride))  # grid size (max stride)
    # # opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples
    #
    # # Input
    # w = model_context[opt.model_key]['w']
    # h = model_context[opt.model_key]['h']
    # img = torch.zeros(1, 3, h, w).to(device)  # image size(1,3,320,192) iDetection
    # # model.model[-1].export = not opt.grid  # set Detect() layer grid export
    # y = yolov5_net(img)  # dry run
    # print(y.shape)
    # exit(0)
    # TorchScript export
    if opt.compact:
        """
        必须pytorch版本一致时得出的pt才能用
        """
        # try:
        #     import onnx
        #     print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        #     f = onnx_weights  # filename
        #     torch.onnx.export(yolov5_net, img, f, verbose=False, opset_version=11, input_names=['images'],
        #                       output_names=['output'])
        #
        #     # Checks
        #     onnx_model = onnx.load(f)  # load onnx model
        #     onnx.checker.check_model(onnx_model)  # check onnx model
        #     # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        #     print('ONNX export success, saved as %s' % f)
        # except Exception as e:
        #     print('ONNX export failure: %s' % e)
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
                        # quantized_algorithm = 'mmse',
                        # mmse_epoch = 6,
                        )
            print('done')

            # Load Pytorch model
            print('--> Loading model')
            dtype = 'int8'

            w = model_context[opt.model_key]['w']
            h = model_context[opt.model_key]['h']

            pre_compile = True if opt.fast else False
            pre_compile_str = 'fast' if pre_compile else 'slow'
            model_pt_full_name = model_context
            model_rknn_full_name = pj('rknn/', '{}_{:d}x{:d}_{}.rknn'.format(opt.model_key,
                                                                                   w,h, pre_compile_str))


            input_size_list = [[3, h, w]]
            # ret = rknn.load_pytorch(model=pt_weights, input_size_list=input_size_list)
            ret = rknn.load_onnx(model=onnx_weights)
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

            print('--> Accuracy analysis')
            rknn.accuracy_analysis(inputs='./dataset.txt', target='rk3399pro')
            print('done')

            # Export RKNN model
            print('--> Export RKNN model')
            ret = rknn.export_rknn(model_rknn_full_name)
            if ret != 0:
                print('Export {} failed!'.format(model_rknn_full_name))
                exit(ret)
            print('done')

            print('--> onnx runtime')
            from utils.onnx_infer import ONNXModel
            import numpy as np

            img = np.ones((1,3,h,w),np.float32)*255
            onnx_model = ONNXModel(onnx_path=onnx_weights)
            output = onnx_model.forward(img)
            if isinstance(output, list):
                for o in output:
                    print(o.shape)
                    print(o[0][:10])
                print(output[-1].reshape(-1))
            else:
                print(output.shape)

            # fast模式下， cpu上不能进行模拟推理
            if not opt.fast:
                print('--> Import RKNN model and infering')
                ret = rknn.load_rknn(model_rknn_full_name)
                # Set inputs
                # img = cv2.imread('./space_shuttle_224.jpg')
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                test_img_path = opt.img
                if test_img_path is None or test_img_path == '':
                    test_img_path = 'test.jpg'
                    print('using default image for test = {}'.format(test_img_path))

                else:
                    print('using image for test = {}'.format(test_img_path))

                import cv2
                img = cv2.imread(test_img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, dsize=(w, h))

                img = np.ones((h, w, 3), np.uint8)*255

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
                for o in outputs:
                    print(o.shape)
                    print(o[0][:10])  # if model already softmax
                print('done')

            rknn.release()

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
