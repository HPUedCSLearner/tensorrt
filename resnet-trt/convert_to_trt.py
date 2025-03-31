import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, trt_engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ONNX 解析失败")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16

    # **新增：优化 profile**
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)

    # 设置动态输入的最小、最优、最大 batch_size
    profile.set_shape(input_tensor.name, min=(1, 3, 224, 224), opt=(8, 3, 224, 224), max=(16, 3, 224, 224))
    config.add_optimization_profile(profile)

    # 生成 TensorRT engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("构建 TensorRT 引擎失败")
    
    with open(trt_engine_path, "wb") as f:
        f.write(serialized_engine)

    print("TensorRT Engine 已保存为", trt_engine_path)
    return serialized_engine

# 执行转换
build_engine("resnet101.onnx", "resnet101.trt")
