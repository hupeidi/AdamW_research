"""Numerical checks that guard the interpretation of the experiments."""
import torch
from train import TinyGPT, decay_groups


def main():
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    g = torch.tensor([0.3, -0.4], dtype=torch.float64)
    old = p.detach().clone()
    p.grad = g.clone()
    opt = torch.optim.AdamW([p], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    opt.step()
    expected = (1 - 0.01 * 0.1) * old - 0.01 * g / (g.abs() + 1e-8)
    torch.testing.assert_close(p, expected, rtol=1e-12, atol=1e-12)
    # With zero decay Adam and AdamW must agree for multiple changing gradients.
    a, w = [torch.nn.Parameter(old.clone()) for _ in range(2)]
    ao = torch.optim.Adam([a], lr=0.01, weight_decay=0, betas=(0.9, 0.95))
    wo = torch.optim.AdamW([w], lr=0.01, weight_decay=0, betas=(0.9, 0.95))
    for i in range(8):
        a.grad = g * (i + 1)
        w.grad = a.grad.clone()
        ao.step()
        wo.step()
    torch.testing.assert_close(a, w, rtol=1e-12, atol=1e-12)
    model = TinyGPT(65, context=16, width=32, layers=2, heads=4, dropout=0).eval()
    x = torch.randint(65, (2, 16))
    y = x.clone()
    y[:, 9:] = (y[:, 9:] + 1) % 65
    torch.testing.assert_close(model(x)[:, :9], model(y)[:, :9], rtol=1e-6, atol=1e-6)
    groups = decay_groups(model, 0.1)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert id(model.token.weight) in {id(p) for p in groups[1]["params"]}
    print("PASS: AdamW formula; zero-decay equivalence; causal mask; parameter grouping.")


if __name__ == "__main__":
    main()
