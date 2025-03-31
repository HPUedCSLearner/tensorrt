## **🚀 在 TensorRT 中使用 FP8 进行推理**

RTX 4060  **支持 FP8** （float8）加速，因此你可以在 **TensorRT 9+** 版本中使用 FP8 进行推理。

---

## **🔥 FP8 支持的 TensorRT 版本**

* **TensorRT 9 及以上** ：官方支持 FP8。
* **TensorRT 8 及以下** ：不支持 FP8，最多支持到 FP16（可以使用 INT8，但没有 FP8 计算单元）。

你的  **RTX 4060（Ada Lovelace 架构）支持 FP8** ，但你需要：

1. **确保 TensorRT 9 已安装** （`trtexec --version` 检查）。
2. **ONNX 模型必须转换为 FP8** （可以使用 `torch.float8_e4m3fn` 量化）。

---

## **✅ 1. 先转换 ResNet-101 ONNX 为 FP8**

### **方式 1：使用 PyTorch 量化 ResNet**

```python
import torch
import torchvision.models as models

# 加载 ResNet-101 预训练模型
model = models.resnet101(pretrained=True)
model.eval()

# 转换权重为 FP8
for param in model.parameters():
    param.data = param.data.to(torch.float8_e4m3fn)  # e4m3 是 FP8 格式之一

# 转换为 ONNX
dummy_input = torch.randn(1, 3, 224, 224).to(torch.float8_e4m3fn)
torch.onnx.export(model, dummy_input, "resnet101_fp8.onnx", opset_version=17)
```

* `float8_e4m3fn`：FP8 格式（比 FP16 更低精度，适用于 Transformer）。
* `opset_version=17`：ONNX 17 版本支持 FP8。

---

## **✅ 2. 使用 TensorRT 9 转换 ONNX 为 FP8**

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_fp8_engine(onnx_path, trt_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX 解析失败")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP8)  # ✅ 启用 FP8
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB 内存

    # 创建优化 profile
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    profile.set_shape(input_tensor.name, (1, 3, 224, 224), (8, 3, 224, 224), (16, 3, 224, 224))
    config.add_optimization_profile(profile)

    # 构建 TensorRT 引擎
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("构建 TensorRT 引擎失败")

    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
    print("FP8 TensorRT Engine 已保存:", trt_path)

build_fp8_engine("resnet101_fp8.onnx", "resnet101_fp8.trt")
```

---

## **✅ 3. 运行 FP8 推理**

```python
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_path):
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("resnet101_fp8.trt")
context = engine.create_execution_context()

# 分配 GPU 内存
input_shape = (1, 3, 224, 224)
d_input = cuda.mem_alloc(int(np.prod(input_shape)) * np.float8_e4m3fn().nbytes)
d_output = cuda.mem_alloc(int(np.prod((1, 1000))) * np.float8_e4m3fn().nbytes)
stream = cuda.Stream()

# 生成随机输入数据
input_data = np.random.randn(*input_shape).astype(np.float8_e4m3fn)

# 复制数据到 GPU
cuda.memcpy_htod_async(d_input, input_data, stream)

# 执行 FP8 推理
context.execute_v2([int(d_input), int(d_output)])

# 复制输出回 CPU
output_data = np.empty((1, 1000), dtype=np.float8_e4m3fn)
cuda.memcpy_dtoh_async(output_data, d_output, stream)
stream.synchronize()

print("FP8 TensorRT 推理完成，输出 shape:", output_data.shape)
```

---

## **🚀 4. 运行 FP8 TensorRT**

```bash
python convert_to_trt.py  # 转换 FP8
python infer_trt.py        # 运行 FP8 推理
```

---

## **🔥 总结**

1. **ONNX 转换** ：

* 使用 **PyTorch 量化** ResNet-101，转换为  **FP8 ONNX** 。

1. **TensorRT 编译** ：

* 使用 `config.set_flag(trt.BuilderFlag.FP8)` 编译 FP8 TensorRT 引擎。

1. **推理** ：

* 确保输入输出 **dtype 为 FP8** (`np.float8_e4m3fn`)。

你可以试试，看看 FP8 能否成功运行，遇到问题随时问我！💪🔥
