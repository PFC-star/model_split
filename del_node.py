"""
create on 2021-4-7 22:48
To Del some unwanted nodes in the OnnxModel
@author: yang
"""

import onnx
import onnx.helper as helper

def Del_node():
    model_path = "Phase4_85_135.onnx"
    model = onnx.load_model(model_path)
    onnx.checker.check_model(model_path)

    # Del Nodes List
    node_name = ["Constant", "Shape", "Gather", "GlobalAveragePool", "Unsqueeze", "Concat", "Reshape", "Gemm", "Sigmoid", "Mul"]
    node_output = ['581', '595']

    graph = model.graph
    print("Node num:", len(graph.node))

    for node in graph.node:
        if str(node.output)[2:5] in node_output:
            print(str(node.output)[2:5])
            model.graph.node.remove(node)

    for i in range(len(graph.node)-1, -1, -1):
        if graph.node[i].op_type in node_name:
            graph.node.remove(graph.node[i])

    # New Output
    model.graph.output.pop(0)
    model.graph.output.append(helper.make_tensor_value_info("568", 1, (1, 2048, 8, 6)))
    model.graph.output[0].name = "568"

    model.graph.output.append(helper.make_tensor_value_info("594", 1, (1, 2048, 8, 6)))
    model.graph.output[1].name = "594"

    print("After del Node num:", len(model.graph.node))
    onnx.save_model(model, "del_model.onnx")

def main():
    Del_node()

if __name__ == '__main__':
    main()
