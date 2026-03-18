"""
Bradley-Terry Model Implementation

This module provides a Bradley-Terry ranking model for pairwise comparisons.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression


class BradleyTerryRanker:
    """
    Simple Bradley-Terry model wrapper using logistic regression.

    The Bradley-Terry model estimates the relative strength of items based on
    pairwise comparison outcomes.
    """

    def __init__(self, scale=400, base=10, init_rating=1000):
        """
        Initialize the Bradley-Terry ranker.

        Args:
            scale: Scale factor for rating conversion
            base: Base for logarithmic scaling
            init_rating: Initial rating value
        """
        self.params = None
        self.model_names = None
        self.model_to_idx = None
        self.scale = scale
        self.base = base
        self.init_rating = init_rating
        self.lr = None
        self.std_errors = None

    def fit(
        self,
        short_uids: List[str],
        models: List[str],
        results_dict: Dict[str, Any],
        verbose: bool = False,
        bootstrap_std: bool = True,
    ):
        """
        Fit the Bradley-Terry model using logistic regression.

        Args:
            short_uids: List of benchmark identifiers
            models: List of model names
            results_dict: Dictionary containing results keyed by full_uid
            verbose: Whether to print verbose output
            bootstrap_std: Whether to compute bootstrap standard errors

        Returns:
            self
        """
        from utils.data_io import parse_verdict

        models = sorted([model.replace("\\/", "/") for model in models])
        self.model_names = models
        self.model_to_idx = {name: idx for idx, name in enumerate(self.model_names)}

        # Convert pairwise comparisons to DataFrame format
        comparisons = []
        for short_uid in short_uids:
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    full_uid = f'{short_uid}_{models[i].replace("/", "_")}_{models[j].replace("/", "_")}'
                    verdict = parse_verdict(full_uid, results_dict, verbose=False)
                    if verdict is not None:
                        (winner, loser), margin = verdict
                        if margin in [">>", ">"]:
                            if models[i] == winner:
                                comparisons.append((models[i], models[j], 1))
                            elif models[j] == winner:
                                comparisons.append((models[i], models[j], 0))
                        elif margin == "=":
                            comparisons.append((models[i], models[j], 2))

        print("len(comparisons)", len(comparisons))
        self.comparisons = pd.DataFrame(comparisons, columns=["model_a", "model_b", "outcome"])

        # Create win matrix from comparisons
        ptbl_a_win = pd.pivot_table(
            self.comparisons[self.comparisons["outcome"] == 1],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )

        # Create loss matrix (model_b wins)
        ptbl_b_win = pd.pivot_table(
            self.comparisons[self.comparisons["outcome"] == 0],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )

        # Create tie matrix
        ptbl_tie = pd.pivot_table(
            self.comparisons[self.comparisons["outcome"] == 2],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )

        # Ensure all models are in all matrices
        all_models = sorted(set(models))
        for ptbl in [ptbl_a_win, ptbl_b_win, ptbl_tie]:
            for model in all_models:
                if model not in ptbl.index:
                    ptbl.loc[model] = 0
                if model not in ptbl.columns:
                    ptbl[model] = 0

        # Sort indices and columns
        ptbl_a_win = ptbl_a_win.loc[all_models, all_models]
        ptbl_b_win = ptbl_b_win.loc[all_models, all_models]
        ptbl_tie = ptbl_tie.loc[all_models, all_models]

        # Combine into full win matrix (including ties as 0.5 wins for each)
        win_matrix = ptbl_a_win + 0.5 * ptbl_tie
        loss_matrix = ptbl_b_win + 0.5 * ptbl_tie
        total_matrix = win_matrix + loss_matrix

        # Prepare data for logistic regression
        X_data = []
        y_data = []
        weights = []

        n_models = len(all_models)
        for i in range(n_models):
            for j in range(n_models):
                if i != j and total_matrix.iloc[i, j] > 0:
                    # Create feature vector: +1 for model i, -1 for model j
                    feature = np.zeros(n_models)
                    feature[i] = 1
                    feature[j] = -1
                    X_data.append(feature)

                    # Win rate as target
                    win_rate = win_matrix.iloc[i, j] / total_matrix.iloc[i, j]
                    y_data.append(1 if win_rate > 0.5 else 0)

                    # Weight by number of comparisons
                    weights.append(total_matrix.iloc[i, j])

        X = np.array(X_data)
        y = np.array(y_data)
        weights = np.array(weights)

        # Fit logistic regression
        self.lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
        self.lr.fit(X, y, sample_weight=weights)

        # Extract parameters (model strengths)
        params = self.lr.coef_[0]

        # Convert to ratings (ELO-like scale)
        ratings = self.init_rating + self.scale * params

        # Create results DataFrame
        self.params = pd.DataFrame({"model": all_models, "rating": ratings, "strength": params})
        self.params = self.params.sort_values("rating", ascending=False).reset_index(drop=True)

        # Compute bootstrap standard errors if requested
        if bootstrap_std:
            self.std_errors = self._bootstrap_std_errors(
                short_uids, models, results_dict, n_bootstrap=100
            )
        else:
            self.std_errors = np.ones(len(self.model_names)) * 0.1

        return self

    def _bootstrap_std_errors(
        self, short_uids: List[str], models: List[str], results_dict: Dict, n_bootstrap: int = 100
    ) -> np.ndarray:
        """
        Compute bootstrap standard errors for model ratings.

        Args:
            short_uids: List of benchmark identifiers
            models: List of model names
            results_dict: Dictionary containing results
            n_bootstrap: Number of bootstrap samples

        Returns:
            Array of standard errors for each model
        """
        from data_loading import parse_verdict

        bootstrap_ratings = []

        for _ in range(n_bootstrap):
            # Resample UIDs with replacement
            resampled_uids = np.random.choice(short_uids, size=len(short_uids), replace=True)

            # Collect comparisons for resampled data
            comparisons = []
            for short_uid in resampled_uids:
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        full_uid = f'{short_uid}_{models[i].replace("/", "_")}_{models[j].replace("/", "_")}'
                        verdict = parse_verdict(full_uid, results_dict, verbose=False)
                        if verdict is not None:
                            (winner, loser), margin = verdict
                            if margin in [">>", ">"]:
                                if models[i] == winner:
                                    comparisons.append((models[i], models[j], 1))
                                elif models[j] == winner:
                                    comparisons.append((models[i], models[j], 0))

            if len(comparisons) == 0:
                continue

            # Fit model on bootstrap sample
            comp_df = pd.DataFrame(comparisons, columns=["model_a", "model_b", "outcome"])

            # Prepare data for logistic regression
            X_data = []
            y_data = []
            weights_data = []

            all_models = sorted(self.model_names)
            n_models = len(all_models)
            model_to_idx = {name: idx for idx, name in enumerate(all_models)}

            for _, row in comp_df.iterrows():
                i = model_to_idx[row["model_a"]]
                j = model_to_idx[row["model_b"]]

                feature = np.zeros(n_models)
                feature[i] = 1
                feature[j] = -1
                X_data.append(feature)
                y_data.append(row["outcome"])
                weights_data.append(1)

            if len(X_data) == 0:
                continue

            X = np.array(X_data)
            y = np.array(y_data)
            weights = np.array(weights_data)

            # Fit logistic regression
            lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
            lr.fit(X, y, sample_weight=weights)

            # Extract ratings
            params = lr.coef_[0]
            ratings = self.init_rating + self.scale * params
            bootstrap_ratings.append(ratings)

        # Compute standard errors
        if len(bootstrap_ratings) > 0:
            bootstrap_ratings = np.array(bootstrap_ratings)
            std_errors = np.std(bootstrap_ratings, axis=0)
        else:
            std_errors = np.ones(len(self.model_names)) * 0.1

        return std_errors

    def get_ratings(self) -> pd.DataFrame:
        """
        Get the fitted ratings as a DataFrame.

        Returns:
            DataFrame with columns: model, rating, strength, std_error
        """
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        df = self.params.copy()
        if self.std_errors is not None:
            # Map std_errors to models in params order
            model_to_stderr = dict(zip(self.model_names, self.std_errors))
            df["std_error"] = df["model"].map(model_to_stderr)

        return df

    def predict_probability(self, model_a: str, model_b: str) -> float:
        """
        Predict the probability that model_a beats model_b.

        Args:
            model_a: Name of first model
            model_b: Name of second model

        Returns:
            Probability that model_a wins
        """
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        i = self.model_to_idx[model_a]
        j = self.model_to_idx[model_b]

        strength_a = self.params.iloc[i]["strength"]
        strength_b = self.params.iloc[j]["strength"]

        # Bradley-Terry probability
        prob = 1 / (1 + np.exp(-(strength_a - strength_b)))
        return prob

    def get_win_matrix(self) -> pd.DataFrame:
        """Get matrix of win probabilities between all pairs."""
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        models = list(self.params["model"])
        n = len(models)
        win_probs = np.zeros((n, n))

        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if i == j:
                    win_probs[i, j] = 0.5
                else:
                    win_probs[i, j] = self.predict_probability(model_a, model_b)

        df = pd.DataFrame(win_probs, index=models, columns=models)
        return df

    def get_empirical_win_matrix(self) -> pd.DataFrame:
        """
        Calculate empirical win rates from the actual comparison data.

        Returns:
            Matrix where entry [i,j] is the observed win rate of model i vs model j
        """
        if not hasattr(self, "comparisons"):
            raise ValueError("No comparison data available. Call fit() first.")

        n = len(self.model_names)
        wins = np.zeros((n, n))
        totals = np.zeros((n, n))

        for _, row in self.comparisons.iterrows():
            model_a, model_b, outcome = row["model_a"], row["model_b"], row["outcome"]
            i = self.model_to_idx[model_a]
            j = self.model_to_idx[model_b]

            totals[i, j] += 1
            totals[j, i] += 1

            if outcome == 1:  # model_a wins
                wins[i, j] += 1
            elif outcome == 0:  # model_b wins
                wins[j, i] += 1
            else:  # tie
                wins[i, j] += 0.5
                wins[j, i] += 0.5

        # Calculate win rates, avoiding division by zero
        win_rates = np.where(totals > 0, wins / totals, 0.5)

        df = pd.DataFrame(win_rates, index=self.model_names, columns=self.model_names)
        return df

    def plot_win_matrix(
        self,
        win_matrix: Optional[pd.DataFrame] = None,
        figsize: tuple = (10, 8),
        cmap: str = "RdYlGn",
        annot: bool = False,
        sort_by_rating: bool = True,
        custom_order: Optional[List[str]] = None,
    ):
        """
        Plot a win rate matrix as a heatmap.

        Args:
            win_matrix: Win rate matrix to plot. If None, uses Bradley-Terry predictions
            figsize: Figure size (width, height)
            cmap: Colormap name (default: 'RdYlGn')
            annot: Whether to annotate cells with values
            sort_by_rating: If True, sort models by their Bradley-Terry rating
            custom_order: Custom order of model names. Overrides sort_by_rating

        Returns:
            matplotlib figure and axis objects
        """
        if win_matrix is None:
            win_matrix = self.get_win_matrix()

        # Determine order
        if custom_order is not None:
            order = [m for m in custom_order if m in win_matrix.index]
        elif sort_by_rating:
            ratings = self.get_ratings()
            order = ratings["model"].tolist()
        else:
            order = win_matrix.index.tolist()

        # Reorder matrix
        win_matrix_sorted = win_matrix.loc[order, order]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            win_matrix_sorted,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            center=0.5,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Win Probability"},
            ax=ax,
        )

        ax.set_title("Model Win Rate Matrix", fontsize=14, pad=20)
        ax.set_xlabel("Opponent Model", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)

        plt.tight_layout()
        return fig, ax


def run_bradley_terry(
    short_uids: List[str],
    models_list: List[str],
    results_dict: Dict,
    verbose: bool = False,
    bootstrap_std: bool = True,
) -> pd.DataFrame:
    """
    Run Bradley-Terry ranking on the given data.

    Args:
        short_uids: List of benchmark identifiers
        models_list: List of model names
        results_dict: Dictionary containing results
        verbose: Whether to print verbose output
        bootstrap_std: Whether to compute bootstrap standard errors

    Returns:
        DataFrame with model rankings
    """
    ranker = BradleyTerryRanker()
    ranker.fit(short_uids, models_list, results_dict, verbose=verbose, bootstrap_std=bootstrap_std)
    return ranker.get_ratings()

