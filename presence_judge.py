import math
from collections import deque
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def judge_presence_from_similarity(
    similarity: torch.Tensor,
    *,
    topk_ratio: float = 0.05,
    active_quantile: float = 0.80,
    active_abs_floor: float = 0.40,
    smooth_kernel: int = 3,
    strength_tau: float = 1.20,
    strength_temp: float = 0.60,
    compactness_tau: float = 0.18,
    # weights: Tuple[float, float, float] = (0.35, 0.30, 0.35),
    total_tau: float = 0.55,
    min_subscores: Optional[Tuple[float, float, float]] = (0.20, 0.20, 0.20),
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Judge whether a target is present from a flattened similarity map.

    Args:
        similarity: Similarity tensor. After ``squeeze()``, it must be 1D.
            Supported examples include shapes ``[N]``, ``[N, 1]``, and
            ``[1, N, 1]``. ``N`` must be a perfect square because the function
            reshapes it into a square 2D map.
        topk_ratio: Ratio of pixels used for the top-k mean. Default: 0.05.
        active_quantile: Quantile threshold over positive score values for the
            active region. Default: 0.80.
        active_abs_floor: Absolute lower bound for the active threshold.
            Default: 0.40.
        smooth_kernel: Average-pooling kernel size for score-map smoothing.
            Values <= 1 disable smoothing. Default: 3.
        strength_tau: Threshold used by the strength sigmoid. Default: 1.20.
        strength_temp: Temperature used by the strength sigmoid. Default: 0.60.
        compactness_tau: Scale used by the compactness exponential score.
            Default: 0.18.
        weights: Weights for strength, compactness, and purity when computing
            the total score. Default: (0.35, 0.30, 0.35).
        total_tau: Minimum total score required for a positive decision.
            Default: 0.55.
        min_subscores: Optional per-subscore lower bounds for strength,
            compactness, and purity. Set to ``None`` to only use ``total_tau``.
            Default: (0.20, 0.20, 0.20).
        eps: Small value for numerical stability. Default: 1e-6.

    Returns:
        A dictionary with:
        - ``exists``: final boolean decision.
        - ``total_score``: weighted score from the three subscores.
        - ``subscores``: strength, compactness, and purity.
        - ``raw_stats``: intermediate scalar diagnostics.
        - ``maps``: raw 2D map, normalized score map, and active mask tensors.
    """

    x = similarity.squeeze()
    if x.dim() != 1:
        raise ValueError(
            f"`similarity` should flatten to 1D after squeeze(), "
            f"got shape={tuple(similarity.shape)}"
        )

    n = x.numel()
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"`similarity` length must be a square number, got N={n}")

    raw_map = x.float().view(side, side)
    flat = raw_map.flatten()

    # Strength checks whether the strongest responses are clearly above the background.
    q50 = torch.quantile(flat, 0.50)
    q95 = torch.quantile(flat, 0.95)

    topk = max(1, int(flat.numel() * topk_ratio))
    topk_mean = torch.topk(flat, k=topk).values.mean()

    strength_raw = torch.clamp((topk_mean - q50) / (q95 - q50 + eps), min=0.0)
    s1 = torch.sigmoid((strength_raw - strength_tau) / strength_temp)

    # The score map is used for compactness and purity.
    score_map = torch.relu((raw_map - q50) / (q95 - q50 + eps))

    if smooth_kernel > 1:
        score_map = F.avg_pool2d(
            score_map[None, None],
            kernel_size=smooth_kernel,
            stride=1,
            padding=smooth_kernel // 2,
        )[0, 0]

    positive_vals = score_map[score_map > 0]

    active_mask = torch.zeros_like(score_map, dtype=torch.bool)
    active_threshold = torch.tensor(0.0, device=score_map.device)
    s2 = torch.tensor(0.0, device=score_map.device)
    s3 = torch.tensor(0.0, device=score_map.device)
    spread = torch.tensor(float("inf"), device=score_map.device)
    largest_cc_size = 0
    active_count = 0

    if positive_vals.numel() > 0:
        quantile_thr = torch.quantile(positive_vals, active_quantile)
        active_threshold = torch.tensor(
            max(float(quantile_thr.item()), float(active_abs_floor)),
            device=score_map.device,
            dtype=score_map.dtype,
        )
        active_mask = score_map >= active_threshold
        active_count = int(active_mask.sum().item())

        if active_count > 0:
            ys, xs = torch.nonzero(active_mask, as_tuple=True)
            weights_active = score_map[ys, xs]

            coords = torch.stack([ys.float(), xs.float()], dim=1)
            center = (weights_active[:, None] * coords).sum(0) / (
                weights_active.sum() + eps
            )

            dists = torch.norm(coords - center[None, :], dim=1)
            h, w = score_map.shape
            diag = math.sqrt((h - 1) ** 2 + (w - 1) ** 2) + eps

            spread = (weights_active * dists).sum() / (
                weights_active.sum() * diag + eps
            )
            s2 = torch.exp(-spread / compactness_tau).clamp(0.0, 1.0)

            def largest_connected_component_energy(
                mask: torch.Tensor,
                weight_map: torch.Tensor,
            ) -> Tuple[float, int]:
                mask_cpu = mask.detach().cpu()
                weight_cpu = weight_map.detach().cpu()

                h, w = mask_cpu.shape
                visited = torch.zeros((h, w), dtype=torch.bool)

                best_energy = 0.0
                best_size = 0

                neighbors = [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]

                for y in range(h):
                    for x in range(w):
                        if (not mask_cpu[y, x]) or visited[y, x]:
                            continue

                        q = deque([(y, x)])
                        visited[y, x] = True
                        cur_energy = 0.0
                        cur_size = 0

                        while q:
                            cy, cx = q.popleft()
                            cur_energy += float(weight_cpu[cy, cx].item())
                            cur_size += 1

                            for dy, dx in neighbors:
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    if mask_cpu[ny, nx] and (not visited[ny, nx]):
                                        visited[ny, nx] = True
                                        q.append((ny, nx))

                        if cur_energy > best_energy:
                            best_energy = cur_energy
                            best_size = cur_size

                return best_energy, best_size

            largest_cc_energy, largest_cc_size = largest_connected_component_energy(
                active_mask, score_map
            )

            total_active_energy = float(score_map[active_mask].sum().item()) + eps
            s3 = torch.tensor(
                largest_cc_energy / total_active_energy,
                device=score_map.device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)

    # w1, w2, w3 = weights
    # total_score = w1 * s1 + w2 * s2 + w3 * s3

    # if min_subscores is None:
    #     exists = bool(total_score >= total_tau)
    # else:
    #     g1, g2, g3 = min_subscores
    #     exists = bool(
    #         (s1 >= g1)
    #         and (s2 >= g2)
    #         and (s3 >= g3)
    #         and (total_score >= total_tau)
    #    )

    return {
        # "exists": exists,
        # "total_score": float(total_score.item()),
        "subscores": {
            "strength": float(s1.item()),
            "compactness": float(s2.item()),
            "purity": float(s3.item()),
        },
        "raw_stats": {
            "q50": float(q50.item()),
            "q95": float(q95.item()),
            "topk_mean": float(topk_mean.item()),
            "strength_raw": float(strength_raw.item()),
            "spread": float(spread.item()) if torch.isfinite(spread) else float("inf"),
            "largest_cc_size": int(largest_cc_size),
            "active_count": int(active_count),
            "active_threshold": float(active_threshold.item()),
        },
        "maps": {
            "raw_map": raw_map,
            "score_map": score_map,
            "active_mask": active_mask,
        },
    }



