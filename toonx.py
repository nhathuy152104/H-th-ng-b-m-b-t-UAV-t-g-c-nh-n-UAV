import torch
from lib.utils.ce_utils import generate_mask_cond
from lib.config.ostrack.config import cfg, update_config_from_file
from lib.models.ostrack.ostrack import build_ostrack


# ===========================
# Load config
# ===========================
update_config_from_file(
    "experiments/ostrack/vitb_384_mae_ce_32x4_ep300.yaml"
)

# ===========================
# Build model
# ===========================
model = build_ostrack(cfg, training=False)

# ===========================
# Load checkpoint
# ===========================
checkpoint = torch.load(
    "Z:/HuyWorkSpace/OSTrack-main/OSTrack-main/output/checkpoints/train/ostrack/vitb_384_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar",
    map_location="cpu",
)

model.load_state_dict(checkpoint["net"], strict=False)
model.eval()

# ===========================
# Dummy input
# ===========================
template = torch.randn(1, 3, 192, 192)
search = torch.randn(1, 3, 384, 384)
ce_template_mask = generate_mask_cond(cfg, 1, template.device, None)
print(template.shape)
print(search.shape)
print(ce_template_mask.shape)
# ===========================
# Test forward
# ===========================
with torch.no_grad():
    out = model(template, search, ce_template_mask)

print(out.keys())

print("\n===== Output Info =====")
for k, v in out.items():
    if isinstance(v, torch.Tensor):
        print(f"{k:20s} Tensor {tuple(v.shape)}")
    else:
        print(f"{k:20s} {type(v)}")


# ===========================
# Wrapper
# ===========================
class OSTrackONNX(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, template, search, ce_template_mask):
        out = self.model(
            template,
            search,
            ce_template_mask=ce_template_mask
        )

        return (
            out["pred_boxes"],
            out["score_map"],
            out["size_map"],
            out["offset_map"],

        )
onnx_model = OSTrackONNX(model)
onnx_model.eval()

# ===========================
# Export
# ===========================
torch.onnx.export(
    onnx_model,
    (template, search, ce_template_mask),
    "ostrack.onnx",
    input_names=[
        "template",
        "search",
        "ce_template_mask"
    ],
    output_names=[
        "pred_boxes",
        "score_map",
        "size_map",
        "offset_map",
    ],
    opset_version=13

)

print("\nExport ONNX Done!")