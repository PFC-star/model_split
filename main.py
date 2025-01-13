import onnx
import numpy as np
from onnx import helper
from onnx import TensorProto
import onnxruntime as rt


def network_construct():
    # 输入层
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 3, 3])  # 输入维度: 1x1x3x3
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 1, 1])  # 输出维度：1x2x1x1（第二个卷积层的输出）

    # 第一层卷积
    conv1_weights = helper.make_tensor(
        name='conv1_weights',
        data_type=TensorProto.FLOAT,
        dims=[4, 1, 2, 2],  # 输出通道4，输入通道1，卷积核大小2x2
        vals=np.random.randn(4 * 1 * 2 * 2).astype(np.float32).flatten().tolist()
    )
    conv1_bias = helper.make_tensor(
        name='conv1_bias',
        data_type=TensorProto.FLOAT,
        dims=[4],  # 偏置大小为4
        vals=np.random.randn(4).astype(np.float32).flatten().tolist()
    )
    conv1_node = helper.make_node(
        'Conv',
        inputs=['X', 'conv1_weights', 'conv1_bias'],
        outputs=['conv1_out'],
        kernel_shape=[2, 2],
        strides=[1, 1],
        pads=[0, 0, 0, 0]
    )

    # 第二层卷积
    conv2_weights = helper.make_tensor(
        name='conv2_weights',
        data_type=TensorProto.FLOAT,
        dims=[2, 4, 2, 2],  # 输出通道2，输入通道4，卷积核大小2x2
        vals=np.random.randn(2 * 4 * 2 * 2).astype(np.float32).flatten().tolist()
    )
    conv2_bias = helper.make_tensor(
        name='conv2_bias',
        data_type=TensorProto.FLOAT,
        dims=[2],  # 偏置大小为2
        vals=np.random.randn(2).astype(np.float32).flatten().tolist()
    )
    conv2_node = helper.make_node(
        'Conv',
        inputs=['conv1_out', 'conv2_weights', 'conv2_bias'],
        outputs=['Y'],  # 第二层卷积的输出连接到最终输出 Y
        kernel_shape=[2, 2],
        strides=[1, 1],
        pads=[0, 0, 0, 0]
    )

    # 创建图
    graph_def = helper.make_graph(
        [conv1_node, conv2_node],  # 两个卷积层
        'simple-cnn',
        [X],  # 模型输入
        [Y],  # 模型输出
        [conv1_weights, conv1_bias, conv2_weights, conv2_bias]  # 模型参数
    )

    # 创建模型
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    # 检查模型
    onnx.checker.check_model(model_def)

    # 保存模型
    onnx.save(model_def, 'simple_cnn.onnx')


def model_infer():
    # 加载模型
    model = onnx.load('simple_cnn.onnx')

    # 创建推理会话
    sess = rt.InferenceSession('simple_cnn.onnx')

    # 模拟输入数据
    input_data = np.random.randn(1, 1, 3, 3).astype(np.float32)

    # 运行推理
    result = sess.run(None, {'X': input_data})

    # 打印输出结果
    print('Inference result:', result)


if __name__ == "__main__":
    network_construct()
    model_infer()