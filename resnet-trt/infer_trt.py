import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from cuda import cudart

# 加载 TensorRT Engine
def load_engine(trt_engine_path):
    with open(trt_engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("resnet101.trt")

# 预分配 GPU 缓存
context = engine.create_execution_context()

context.set_input_shape('input', (2, 3, 224, 224))  # 设置输入形状

input_data = np.random.randn(2, 3, 224, 224).astype(np.float32)

d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(2 * 1000 * np.float32().nbytes)  # ResNet-101 有 1000 个分类


cuda.memcpy_htod(d_input, input_data)  # 将输入数据复制到 GPU


context.set_tensor_address('input', int(d_input))  # 设置输入张量地址
context.set_tensor_address('output', int(d_output))  # 设置输出张量地址

_, stream = cudart.cudaStreamCreate()  # 创建 CUDA 流


context.execute_async_v3(stream_handle=stream)  # 异步执行推理

cudart.cudaStreamSynchronize(stream)  # 等待流完成
cudart.cudaStreamDestroy(stream)  # 销毁流


output_data = np.empty((2, 1000), dtype=np.float32)  # 创建输出数据数组
cuda.memcpy_dtoh(output_data, d_output)  # 将输出数据从 GPU 复制到 CPU

print("TensorRT 推理完成，输出 shape:", output_data.shape)

print("输出数据:", output_data[0][:10])
print("输出数据:", output_data[1][:10])

# 释放 GPU 缓存
# d_input.free()
# d_output.free()


import time

# PyTorch 计时
# model.eval()
# x = torch.randn(1, 3, 224, 224)

# start = time.time()
# for _ in range(100):
#     _ = model(x)
# end = time.time()
# print(f"PyTorch 100 次推理时间: {end - start:.4f} 秒")

# TensorRT 计时
# start = time.time()
# for _ in range(100):
#     context.execute_v2([int(d_input), int(d_output)])
# end = time.time()
# print(f"TensorRT 100 次推理时间: {end - start:.4f} 秒")



