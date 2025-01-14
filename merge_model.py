import time
import onnx
from onnx import helper, shape_inference
import onnxruntime as rt
import numpy as np


def merge_onnx_models(model1_path, model2_path, merged_model_path):
    """
    将两个按层拆分的 ONNX 模型合并为一个模型
    :param model1_path: 第一个模型的路径
    :param model2_path: 第二个模型的路径
    :param merged_model_path: 合并后的模型保存路径
    """
    # 加载两个模型
    model1 = onnx.load(model1_path)
    model2 = onnx.load(model2_path)

    # 获取两个模型的计算图
    graph1 = model1.graph
    graph2 = model2.graph
    for output in graph1.output:
        output_name = output.name
    # 更新 graph2 的输入节点名称为 graph1 的输出
    for input in graph2.input:
        if input.name == output_name:  # 这里确保第二个模型的输入与第一个模型的输出对应
            input.type.tensor_type.shape.dim[0].dim_value = graph1.output[0].type.tensor_type.shape.dim[0].dim_value

    # 将 graph1 的节点、初始化器、输入添加到新的图中
    merged_nodes = list(graph1.node) + list(graph2.node)
    merged_initializers = list(graph1.initializer) + list(graph2.initializer)

    # 使用 graph1 的输入和 graph2 的输出作为新模型的输入和输出
    merged_inputs = graph1.input
    merged_outputs = graph2.output

    # 创建新的计算图
    merged_graph = helper.make_graph(
        nodes=merged_nodes,
        name="MergedModel",
        inputs=merged_inputs,
        outputs=merged_outputs,
        initializer=merged_initializers
    )

    # 创建新的模型
    merged_model = helper.make_model(merged_graph)
    merged_model = shape_inference.infer_shapes(merged_model)  # 推断形状

    # 保存合并后的模型
    onnx.save(merged_model, merged_model_path)
    # print(f"合并后的模型已保存到: {merged_model_path}")


def infer_onnx_model(model_path, input_data, warmup=True, runs=10):
    """
    使用 ONNX Runtime 对模型进行推理，并优化时间测量
    """
    # 加载模型并记录加载时间
    start_load = time.time()
    session = rt.InferenceSession(model_path)
    end_load = time.time()
    load_time = end_load - start_load

    # 获取模型的输入和输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 预热模型
    if warmup:
        result = session.run([output_name], {input_name: input_data})

    # 多次推理测量
    durations = []
    for _ in range(runs):
        start = time.time()
        session.run([output_name], {input_name: input_data})
        end = time.time()
        durations.append(end - start)
        # print(end - start)
    average_time = sum(durations)/len(durations)
    return load_time,average_time,result[0]
def inference(model,input_data):

    load_time,average_time,result = infer_onnx_model(model, input_data)

    print(f"加载时间: {load_time:.8f}s, 平均推理时间: {average_time:.8f}s")
    print(f"总时间（加载+推理）: { average_time + load_time:.8f}s\n")
    # print(model_output)
def inference_split(model1,model2,input_data):
    load_time1, average_time1, result = infer_onnx_model(model1, input_data)
    load_time2, average_time2, result = infer_onnx_model(model2, result)

    print(f"加载时间: {load_time1+load_time2:.8f}s, 平均推理时间: {average_time1+average_time2:.8f}s")
    print(f"总时间（加载+推理）: {load_time1+load_time2+ average_time1+average_time2:.8f}s\n")
def inference_merge(trimmed_layer1_model,trimmed_layer2_model,input_data):
    merged_model = "merged_model.onnx"
    starttime = time.time()
    # 合并模型
    merge_onnx_models(trimmed_layer1_model, trimmed_layer2_model, merged_model)
    endtime = time.time()
    print("merge time :", endtime - starttime, " s")
    load_time, average_time, result = infer_onnx_model(merged_model, input_data)

    print(f"加载时间: {load_time:.8f}s, 平均推理时间: {average_time:.8f}s")
    print(f"总时间（merge+加载+推理）: { endtime - starttime + average_time+load_time:.8f}s")

    # print(model_output)
def main():
    # 模型路径
    trimmed_layer1_model = "trimmed_hidden_layer_1.onnx"
    trimmed_layer2_model = "trimmed_hidden_layer_2.onnx"
    initial_model = "simple_cnn.onnx"
    # 定义合并后的模型路径


    # 创建随机输入数据
    input_data = np.random.randn(1, 1, 3, 3).astype(np.float32)
    print("完整模型一次性推理")
    inference(initial_model, input_data)
    print()
    # # 合并模型
    # merge_onnx_models(trimmed_layer1_model, trimmed_layer2_model, merged_model)



    # 合并后推理
    print("模型分割合并后一次性推理")
    inference_merge(trimmed_layer1_model, trimmed_layer2_model, input_data)


    print()
    # 完整模型整块推理
    print("完整模型一次性推理")
    inference(initial_model, input_data)

    # 串行推理

    print("模型分割后串行推理")
    inference_split(trimmed_layer1_model,trimmed_layer2_model,input_data)

    print()






if __name__ == "__main__":
    main()