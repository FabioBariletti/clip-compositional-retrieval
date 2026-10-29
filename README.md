# Dynamic and Hybrid Conditioning for Compositional Image Retrieval

**Fabio Bariletti, Sara Francavilla** — Deep Learning course project, University of Trento (2026)

Compositional image retrieval on **CelebA**: given a reference face image and a set of
positive (`+`) and negative (`−`) attribute constraints (e.g. `+ Eyeglasses & − Smiling`),
retrieve images that keep the identity of the reference while satisfying every constraint.

The project builds on CLIP ViT-B/32 and on the geometric conditioning ideas of
**CLAY** (Lim et al., CVPR 2026) and **GDE** (Berasi et al., CVPR 2025), replacing the
naïve pre-SVD stacking of conditions with a learned fusion module.

The whole project — code and report — lives in a single Colab notebook:
[`DL_project_FinalVersion.ipynb`](DL_project_FinalVersion.ipynb).

---

## Method overview

The composite query embedding ("pointer") is built as

```
pointer = normalize( α · v_ref + (1 − α) · ( a · µ_pos − b · µ_neg + c · µ_contr ) )
```

| Term | Meaning |
|------|---------|
| `v_ref` | CLIP image embedding of the reference |
| `µ_pos` | mean text embedding of the top‑3 synonyms of each positive attribute |
| `µ_neg` | mean text embedding of the top‑3 synonyms of each negative attribute (direction to move away from) |
| `µ_contr` | mean text embedding of the top‑3 contraries of each negative attribute (direction to move toward) |
| `a, b, c` | weights controlling each conditioning direction |

Synonyms/contraries come from a precomputed pool of 10 + 10 prompts per CelebA attribute
([`Syn-Contr.json`](Syn-Contr.json)); CLIP itself is used to select the 3 most discriminative ones.

The notebook explores, in order:

1. **Zero-shot baseline** — plain CLIP latent arithmetic `v_ref + t_pos − t_neg`.
2. **Global optimizer** — learns a single set of scalars `(a, b, c)` with a hard-negative
   margin ranking loss.
3. **CLAY pipeline** — tangent-space log maps, SVD text subspaces `P_c = V_k V_kᵀ`,
   rotation of visual features, and a weighted combination of per-condition projections.
4. **DynamicWeightMLP** (final model) — a small MLP (2048 → 256 → 128 → 3, softplus output)
   that predicts `(a, b, c)` *per query and per reference image* from
   `[µ_pos | µ_neg | µ_contr | v_ref]`.

Training pairs are generated from the CelebA attribute labels with the same relaxed
Hamming ≤ 2 criterion as the benchmark (200 references per query); the official
evaluation JSON is used only for testing.

## Results (CelebA test split, Recall@K averaged over queries)

| Model | R@1 | R@5 | R@10 |
|-------|----:|----:|-----:|
| Zero-shot baseline | 2.23% | 6.97% | 10.46% |
| **DynamicWeightMLP** | 2.18% | **8.44%** | **14.33%** |

The largest gains are on composed queries, e.g. `− Smiling & + Eyeglasses & + Wearing Hat`
(R@10 8.86% → 34.18%) and `+ Eyeglasses` (12.07% → 25.73%). The global optimizer and the
CLAY-based variant both stay below the baseline; the per-query tables, Precision@K,
qualitative examples and a full discussion are in the notebook.

## Repository structure

```
.
├── DL_project_FinalVersion.ipynb   # code + report (run on Google Colab, T4 GPU)
├── Syn-Contr.json                  # synonyms / contraries pool for the 40 CelebA attributes
├── Sin-Contr_2.json                # alternative pool of full-sentence prompts (not used by the final notebook)
├── baseline/
│   └── baseline_results.json       # cached zero-shot baseline metrics
├── requirements.txt
└── README.md
```

## Running the notebook

The notebook is designed for **Google Colab** and reads/writes everything from a Google
Drive folder (`DRIVE_PROJECT_PATH`, default `/content/drive/MyDrive/DeepLearning_Project`).
Expected Drive layout:

```
DeepLearning_Project/
├── dataset/
│   ├── celeba.zip                  # CelebA (torchvision layout), not included here
│   └── celeba_evaluation.json      # official benchmark queries, provided by the course
├── Syn-Contr.json                  # from this repo
├── baseline/                       # created / reused automatically
├── embeddings/                     # CLIP image embeddings, built on first run
└── ordered_ann.js                  # ordered ground truth, built on first run
```

1. Copy the files of this repository into the Drive folder above.
2. Add `celeba.zip` and `celeba_evaluation.json` to `dataset/`.
3. Open the notebook in Colab with a GPU runtime and run all cells.

CelebA images, the course evaluation JSON and the cached embeddings are not tracked in
this repository (see [`.gitignore`](.gitignore)).

## References

- D. Berasi, M. Farina, M. Mancini, E. Ricci, N. Strisciuglio. *Not only text: Exploring compositionality of visual representations in vision-language models.* CVPR 2025.
- S. Lim, L. Hyoseok, J. Park, T.-H. Oh. *CLAY: Conditional visual similarity modulation in vision-language embedding space.* CVPR 2026.
- A. Radford et al. *Learning transferable visual models from natural language supervision.* ICML 2021.
- Z. Liu, P. Luo, X. Wang, X. Tang. *Deep learning face attributes in the wild.* ICCV 2015.
