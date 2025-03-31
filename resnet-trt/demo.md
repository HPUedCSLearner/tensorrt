 **ResNet-101** ，在 **RTX 4060** 上进行 **ONNX → TensorRT 转换** 并进行推理

### **任务流程**

✅ 加载 PyTorch 预训练的 **ResNet-101**

✅  **转换为 ONNX** （支持动态输入）

✅  **使用 TensorRT 进行转换** （`trtexec` 或 TensorRT Python API）

✅ **TensorRT 进行推理并对比 PyTorch**

---

### **依赖环境**

确保你已安装：

```bash
pip install torch torchvision onnx onnxruntime
pip install tensorrt
```

（如果 `tensorrt` 不能直接安装，需去 NVIDIA 官网下载）

---

### **完整代码**

#### **1️⃣ PyTorch → ONNX**

```python
import torch
import torchvision.models as models

# 加载预训练的 ResNet-101
model = models.resnet101(pretrained=True)
model.eval()

# 创建输入张量（batch_size=1, 3通道, 224x224）
dummy_input = torch.randn(1, 3, 224, 224)

# 导出 ONNX 模型
torch.onnx.export(
    model, 
    dummy_input, 
    "resnet101.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # 允许动态 batch
    opset_version=17  # 选择较新的 ONNX 版本
)

print("ONNX 模型已导出为 resnet101.onnx")
```

---

#### **2️⃣ ONNX → TensorRT**

使用 `trtexec`（命令行转换），或 Python API 转换。

✅ **方式 1：使用 `trtexec`（推荐）**

```bash
trtexec --onnx=resnet101.onnx --saveEngine=resnet101.trt --fp16
```

✅ **方式 2：Python API**

```python
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
    config.set_flag(trt.BuilderFlag.FP16)  # 使用 FP16 提升速度

    engine = builder.build_engine(network, config)
    with open(trt_engine_path, "wb") as f:
        f.write(engine.serialize())

    print("TensorRT Engine 已保存为", trt_engine_path)
    return engine

# 执行转换
build_engine("resnet101.onnx", "resnet101.trt")
```

---

#### **3️⃣ TensorRT 进行推理**

```python
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 加载 TensorRT Engine
def load_engine(trt_engine_path):
    with open(trt_engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("resnet101.trt")

# 预分配 GPU 缓存
context = engine.create_execution_context()
input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)  # ResNet-101 有 1000 个分类

d_input = cuda.mem_alloc(np.prod(input_shape) * np.float32().nbytes)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.float32().nbytes)
stream = cuda.Stream()

# 生成随机输入
input_data = np.random.randn(*input_shape).astype(np.float32)

# 复制数据到 GPU
cuda.memcpy_htod_async(d_input, input_data, stream)

# 执行推理
context.execute_v2([int(d_input), int(d_output)])

# 复制输出回 CPU
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output_data, d_output, stream)
stream.synchronize()

print("TensorRT 推理完成，输出 shape:", output_data.shape)
```

---

### **4️⃣ PyTorch vs TensorRT 性能对比**

```python
import time

# PyTorch 计时
model.eval()
x = torch.randn(1, 3, 224, 224)

start = time.time()
for _ in range(100):
    _ = model(x)
end = time.time()
print(f"PyTorch 100 次推理时间: {end - start:.4f} 秒")

# TensorRT 计时
start = time.time()
for _ in range(100):
    context.execute_v2([int(d_input), int(d_output)])
end = time.time()
print(f"TensorRT 100 次推理时间: {end - start:.4f} 秒")
```

---

### **结果（示例）**

```
PyTorch 100 次推理时间: 3.21 秒
TensorRT 100 次推理时间: 0.85 秒
```

在 **RTX 4060** 上，TensorRT 可能比 PyTorch 快  **3-5 倍** 。

---

### **总结**

✅ 成功将 **ResNet-101** 从 **PyTorch → ONNX → TensorRT**

✅  **TensorRT 推理速度提升** ，适合部署

✅  **支持 FP16 加速** ，进一步优化推理速度

你可以运行这段代码试试看，如果遇到问题，随时问我！🚀
