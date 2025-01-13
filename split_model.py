import onnx
import onnxruntime as rt
import numpy as np
from onnx import helper, shape_inference


def trim_onnx_model_with_input(model_path, output_path, input_name, input_shape, output_name):
    """
    裁剪 ONNX 模型，只保留从指定的输入到指定的输出路径的计算图，并添加输入节点。
    :param model_path: 原始模型路径
    :param output_path: 保存裁剪后的模型路径
    :param input_name: 指定的输入节点名称
    :param input_shape: 指定的输入张量形状
    :param output_name: 指定的输出节点名称
    """
    # 加载原始 ONNX 模型
    model = onnx.load(model_path)

    # 查找涉及到的节点
    involved_nodes = []
    involved_initializers = set()
    for node in model.graph.node:
        if output_name in node.output:  # 找到目标输出节点
            involved_nodes.append(node)
            current_inputs = set(node.input)
            involved_initializers.update(node.input)
            # 逆向追踪，查找所有相关节点
            while current_inputs:
                current_input = current_inputs.pop()
                for n in model.graph.node:
                    if current_input in n.output:
                        if current_input == input_name:  # 如果当前节点已经是输入节点，停止追踪
                            continue
                        involved_nodes.append(n)
                        current_inputs.update(n.input)
                        # 添加到需要的 initializer
                        involved_initializers.update(n.input)

    # 保留所有与节点相关的 initializer
    new_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name in involved_initializers
    ]

    # 检查所有涉及的节点是否正确关联权重
    for node in involved_nodes:
        for name in node.input:
            if name not in [initializer.name for initializer in new_initializers]:
                print(f"警告：节点 {node.name} 的输入 {name} 未关联到任何权重！")

    # 添加输入节点
    input_tensor = helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)

    # 添加输出节点
    output_tensor = helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None)

    # 创建新的图
    new_graph = helper.make_graph(
        nodes=involved_nodes,
        name="trimmed_model",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=new_initializers
    )

    # 创建新的模型
    trimmed_model = helper.make_model(new_graph)
    trimmed_model = shape_inference.infer_shapes(trimmed_model)

    # 保存裁剪后的模型
    onnx.save(trimmed_model, output_path)
    print(f"裁剪后的模型已保存到: {output_path}")
def infer_onnx_model(model_path, input_data):
    """
    使用 ONNX Runtime 对模型进行推理。
    :param model_path: ONNX 模型路径
    :param input_data: 输入数据
    :return: 模型输出
    """
    # 加载裁剪后的模型
    session = rt.InferenceSession(model_path)

    # 获取模型的输入和输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 执行推理
    result = session.run([output_name], {input_name: input_data})
    return result[0]


def main():
    # 定义模型路径
    original_model = "simple_cnn.onnx"
    trimmed_layer1_model = "trimmed_hidden_layer_1.onnx"
    trimmed_layer2_model = "trimmed_hidden_layer_2.onnx"

    # 裁剪模型，只保留隐藏层 1 的部分
    trim_onnx_model_with_input(
        model_path=original_model,
        output_path=trimmed_layer1_model,
        input_name="X",          # 隐藏层 1 的输入
        input_shape=[1, 1, 3, 3], # 输入张量的形状
        output_name="conv1_out"  # 隐藏层 1 的输出
    )

    # 裁剪模型，只保留隐藏层 2 的部分
    trim_onnx_model_with_input(
        model_path=original_model,
        output_path=trimmed_layer2_model,
        input_name="conv1_out",  # 隐藏层 2 的输入
        input_shape=[1, 4, 2, 2], # 输入张量的形状
        output_name="Y"          # 隐藏层 2 的输出
    )

    # 创建随机输入数据
    input_data = np.random.randn(1, 1, 3, 3).astype(np.float32)

    # 推理隐藏层 1
    print("推理隐藏层 1...")
    layer1_output = infer_onnx_model(trimmed_layer1_model, input_data)
    print("隐藏层 1 输出:")
    print(layer1_output)

    # 推理隐藏层 2
    print("\n推理隐藏层 2...")
    layer2_output = infer_onnx_model(trimmed_layer2_model, layer1_output)
    print("隐藏层 2 输出 (最终输出):")
    print(layer2_output)

    model_output = infer_onnx_model(original_model,input_data)
    print("模型输出 (最终输出):")
    print(model_output)



if __name__ == "__main__":
    main()