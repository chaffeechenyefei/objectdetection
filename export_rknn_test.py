from rknn.api import RKNN
QUANTIZE_ON = False

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


model_rknn_full_name = 'rknn_test.rknn'
onnx_weights = '/project/rk3399_workspace/rknpu_firefly/rknn/rknn_api/examples/rknn_yolov5_demo/convert_rknn_demo/yolov5/onnx_models/yolov5s_rm_transpose.onnx'

# ret = rknn.load_pytorch(model=pt_weights, input_size_list=input_size_list)
ret = rknn.load_onnx(model=onnx_weights)
if ret != 0:
    print('Load Pytorch model failed!')
    exit(ret)
print('done')
print('--> Building model')

ret = rknn.build(do_quantization=True, dataset='./datasets.txt', pre_compile=False)
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
rknn.release()