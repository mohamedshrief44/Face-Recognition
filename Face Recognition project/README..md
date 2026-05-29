# Face Recognition using PCA & LDA

Face recognition on the **ORL (AT&T) dataset** (400 images, 40 subjects × 10) using classical dimensionality-reduction + K-NN classification.

## Algorithms
- **PCA** — economy SVD trick (200×200 covariance), variance threshold α ∈ {0.80, 0.85, 0.90, 0.95}
- **Multiclass LDA** — PCA pre-processing to solve Small Sample Size problem, Sₒ regularisation (λ=1e-4), 39 components
- **Binary LDA** — face vs non-face, 1 eigenvector (rank(Sᵦ)=1)
- **K-NN** — K ∈ {1, 3, 5, 7}, tie-break by mean distance

## Libraries
`numpy` · `scikit-learn` · `matplotlib` · `scipy` · `skimage`

## Key Results

| Method | Split | K | Accuracy |
|---|---|---|---|
| PCA α=0.80 (r=36) | 5/5 | 1 | **95.00%** |
| Multiclass LDA | 5/5 | 1 | 59.50% |
| PCA α=0.85 (r=57) | 7/3 | 1 | **96.67%** |
| Multiclass LDA | 7/3 | 1 | **82.50%** |
| Binary LDA (face/non-face) | — | 1 | **97–98%** |

## Insights
- Higher α in PCA **hurts** accuracy (noise overfitting); sweet spot is α=0.80–0.85
- LDA underperforms with 5 samples/class but gains **+23%** with 7 samples, proving it's a data problem not a method flaw
- K=1 optimal for PCA; K=3 marginally better for LDA on noisy projections

## Dataset Setup
Dataset is **not included** in this repo. Download the ORL (AT&T) Face Database:

🔗 https://cam-orl.co.uk/facedatabase.html

Extract and place under:
```
data/
└── orl_faces/
    ├── s1/   (10 images: 1.pgm … 10.pgm)
    ├── s2/
    └── ... s40/
```
400 images total · 92×112 px grayscale · 10,304 features/image
