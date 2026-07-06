import onnx
from onnxconverter_common import float16

model = onnx.load("ostrack.onnx")

model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True,
    disable_shape_infer=True,
    op_block_list=[
        "Add",
        "LayerNormalization",
        "Softmax",
        "MatMul",
        "Cast",
        "ReduceMean"
    ]
)

onnx.save(
    model_fp16,
    "ostrack_fp16.onnx"
)

print("done")