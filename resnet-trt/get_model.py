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