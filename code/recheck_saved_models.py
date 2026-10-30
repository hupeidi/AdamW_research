"""Independently recompute test CE from the locally produced final checkpoints."""
import json
import torch
from train import ROOT, TinyGPT, evaluate, load_data, write_json


def main():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, meta = load_data(device)
    checked = []
    for path in sorted((ROOT / "results" / "main").glob("*/final_model.pt")):
        # These are this project's own locally generated files, never untrusted downloads.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["vocab"] == meta["vocab"]
        model = TinyGPT(len(payload["vocab"]), **payload["model_config"]).to(device)
        model.load_state_dict(payload["model"])
        loss, tokens = evaluate(model, data["test"], device)
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        delta = abs(loss - summary["test_ce"])
        assert delta < (2e-4 if device.type == "cuda" else 0.02), (path, delta)
        checked.append(dict(run=path.parent.name, recomputed_test_ce=loss,
                            saved_test_ce=summary["test_ce"], absolute_difference=delta, tokens=tokens))
    assert len(checked) == 18, "Expected all 18 final models."
    write_json(ROOT / "results" / "checkpoint_verification.json", dict(device=str(device), runs=checked))
    print(f"PASS: independently evaluated {len(checked)} saved models; "
          f"max absolute CE difference = {max(r['absolute_difference'] for r in checked):.3g}")


if __name__ == "__main__":
    main()
