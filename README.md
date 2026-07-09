# Movie Recommender — Numerical Analysis Project

A small recommender system built as a **Numerical Analysis** project. It predicts
movie ratings using **low-rank matrix factorization** and compares different
numerical approaches:

- a **global-mean baseline**,
- **truncated SVD** on the rating matrix (and why naively filling missing entries
  fails),
- **matrix factorization** trained with **SGD** and with **ADAM**, compared on
  convergence speed and robustness to the learning rate.

It closes with a practical demo: top-N recommendations for a user and
"similar movies" via cosine similarity in the latent space.

The dataset is [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/)
(≈100k ratings, 610 users, 9724 movies).

## Repository structure

```
numerical-project/
├── ml-latest-small/      # dataset (ratings.csv, movies.csv)
├── notebook.ipynb        # the full project
├── requirements.txt      # Python dependencies
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- The packages listed in `requirements.txt` (numpy, pandas, matplotlib,
  scikit-learn, jupyter)

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Open the notebook and run all cells:

```bash
jupyter notebook notebook.ipynb
```

Then use **Kernel → Restart & Run All**. The notebook runs top to bottom in a
couple of minutes and produces all tables and plots inline.

## Data

The notebook reads the CSV files from the `ml-latest-small/` folder using
relative paths, so keep that folder next to `notebook.ipynb`. If it is missing,
download `ml-latest-small.zip` from
[GroupLens](https://grouplens.org/datasets/movielens/) and unzip it here.

## Results (summary)

| Method                        | Test RMSE |
|-------------------------------|-----------|
| Global-mean baseline          | 1.0488    |
| SVD, zero-filled              | 3.0233    |
| SVD, mean-filled              | ~1.00     |
| Matrix factorization (SGD)    | 0.8992    |

Lower is better. Matrix factorization clearly outperforms the baseline, while a
naive zero-filled SVD does not — the notebook explains why.

### Optimizer comparison: SGD vs ADAM

The matrix factorization is trained with both **SGD** and **ADAM**. The point of
this comparison is *not* which one reaches a lower final RMSE, but their
convergence behaviour and robustness to the learning rate:

| Optimizer (lr = 0.02) | Final RMSE | Epochs to RMSE < 0.95 | Relative time |
|-----------------------|------------|-----------------------|---------------|
| SGD                   | 0.8992     | 6                     | ~1×           |
| ADAM                  | 0.9305     | 11                    | ~10×          |

At this learning rate plain SGD is faster and slightly more accurate. ADAM's
advantage shows up across the whole learning-rate sweep: its final RMSE stays
stable over a wide range of step sizes, whereas SGD degrades or diverges at the
extremes. The notebook illustrates this with the learning-rate sensitivity plots.
