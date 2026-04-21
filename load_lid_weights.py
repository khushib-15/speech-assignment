"""
Loader script for best_lid_model.pt
Usage:
    from load_lid_weights import load_lid_model
    model = load_lid_model("best_lid_model.pt")
    model.eval()
"""

import torch
import torch.nn as nn
import struct, pickle, zipfile, io


class MultiHeadLID(nn.Module):
    """Exact architecture from speech_A2.ipynb"""
    def __init__(self, input_dim=120, hidden_dim=256, num_layers=3,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout
        )
        self.layer_norm   = nn.LayerNorm(hidden_dim * 2)
        self.dropout      = nn.Dropout(dropout)
        self.frame_head   = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
        self.segment_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _         = self.bilstm(x)
        out            = self.layer_norm(out)
        out            = self.dropout(out)
        frame_logits   = self.frame_head(out)
        attn           = torch.softmax(self.attention(out), dim=1)
        seg_repr       = (attn * out).sum(dim=1)
        segment_logits = self.segment_head(seg_repr)
        return frame_logits, segment_logits


def load_lid_model(pt_path: str, device: str = "cpu") -> MultiHeadLID:
    """
    Load MultiHeadLID from best_lid_model.pt.
    Works whether the file was saved by this repo or by torch.save().
    """
    model = MultiHeadLID().to(device)

    try:
        # Try standard torch.load first (if saved with torch.save)
        sd = torch.load(pt_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
            model.load_state_dict(sd)
            print("[load_lid_model] Loaded via torch.load (tensor state dict)")
            return model
    except Exception:
        pass

    # Fallback: our custom zip+pickle format (lists of floats → tensors)
    with zipfile.ZipFile(pt_path, "r") as zf:
        with zf.open("archive/data.pkl") as f:
            packed = pickle.load(f)

    sd = {}
    for k, v in packed.items():
        n = len(v) // 4
        floats = list(struct.unpack(f"<{n}f", v))
        # infer shape from model's own parameter
        param = dict(model.named_parameters())[k]
        t = torch.tensor(floats, dtype=torch.float32).reshape(param.shape)
        sd[k] = t

    model.load_state_dict(sd)
    print("[load_lid_model] Loaded via custom zip loader")
    return model


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "best_lid_model.pt"
    m = load_lid_model(path)
    total = sum(p.numel() for p in m.parameters())
    print(f"Model loaded successfully | Parameters: {total:,}")
    # Quick forward pass test
    x = torch.randn(1, 200, 120)
    with torch.no_grad():
        fl, sl = m(x)
    print(f"Frame logits shape  : {tuple(fl.shape)}  (expected [1, 200, 2])")
    print(f"Segment logits shape: {tuple(sl.shape)}  (expected [1, 2])")
    print("Forward pass OK")
