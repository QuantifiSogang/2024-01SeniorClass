import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import OAS
class RiskMetrics:
    def __init__(self):
        return

    @staticmethod
    def calculate_variance(covariance, weights):
        return np.dot(weights, np.dot(covariance, weights))

    @staticmethod
    def calculate_value_at_risk(returns, confidence_level=0.05):
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)

        return returns.quantile(confidence_level, interpolation='higher')[0]

    def calculate_expected_shortfall(self, returns, confidence_level=0.05):
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)

        value_at_risk = self.calculate_value_at_risk(returns, confidence_level)
        expected_shortfall = np.nanmean(returns[returns < value_at_risk])
        return expected_shortfall

    @staticmethod
    def calculate_conditional_drawdown_risk(returns, confidence_level=0.05):
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)

        drawdown = returns.expanding().max() - returns
        max_drawdown = drawdown.expanding().max()
        max_drawdown_at_confidence_level = max_drawdown.quantile(confidence_level, interpolation='higher')
        conditional_drawdown = np.nanmean(max_drawdown[max_drawdown > max_drawdown_at_confidence_level])
        return conditional_drawdown
class ReturnEstimation:
    def __init__(self) -> None :
        return
    @staticmethod
    def mean_historical_return_calculator(asset_prices, resample_by = None, frequency : int = 252):
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.mean() * frequency
        return returns
    @staticmethod
    def exponential_historical_return_calculator(asset_prices, resample_by = None,
                                                 frequency : int = 252, span : int = 500):
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.ewm(span = span).mean().iloc[-1] * frequency
        return returns
    @staticmethod
    def return_calculator(asset_prices, resample_by = None):
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        asset_returns = asset_prices.pct_change()
        asset_returns = asset_returns.dropna(how='all')
        return asset_returns
class HierarchicalRiskParity:
    def __init__(self):
        self.weights = list()
        self.seriated_correlations = None
        self.seriated_distances = None
        self.ordered_indices = None
        self.clusters = None
        self.returns_estimator = ReturnEstimation()
        self.risk_metrics = RiskMetrics()

    @staticmethod
    def _tree_clustering(correlation, method='single'):
        distances = np.sqrt((1 - correlation).round(5) / 2)
        clusters = linkage(squareform(distances.values), method=method)
        return distances, clusters

    def _quasi_diagnalization(self, num_assets, curr_index):
        if curr_index < num_assets:
            return [curr_index]

        left = int(self.clusters[curr_index - num_assets, 0])
        right = int(self.clusters[curr_index - num_assets, 1])

        return (self._quasi_diagnalization(num_assets, left) + self._quasi_diagnalization(num_assets, right))

    def _get_seriated_matrix(self, assets, distances, correlations):
        ordering = assets[self.ordered_indices]
        seriated_distances = distances.loc[ordering, ordering]
        seriated_correlations = correlations.loc[ordering, ordering]
        return seriated_distances, seriated_correlations

    @staticmethod
    def _get_inverse_variance_weights(covariance):
        inv_diag = 1 / np.diag(covariance.values)
        parity_w = inv_diag * (1 / np.sum(inv_diag))
        return parity_w

    def _get_cluster_variance(self, covariance, cluster_indices):
        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        cluster_variance = self.risk_metrics.calculate_variance(covariance=cluster_covariance, weights=parity_w)
        return cluster_variance

    def _recursive_bisection(self, covariance, assets):
        self.weights = pd.Series(1, index=self.ordered_indices)
        clustered_alphas = [self.ordered_indices]

        while clustered_alphas:
            clustered_alphas = [cluster[start:end]
                                for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]

            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]

                left_cluster_variance = self._get_cluster_variance(covariance, left_cluster)
                right_cluster_variance = self._get_cluster_variance(covariance, right_cluster)
                alloc_factor = 1 - left_cluster_variance / (left_cluster_variance + right_cluster_variance)

                self.weights[left_cluster] *= alloc_factor
                self.weights[right_cluster] *= 1 - alloc_factor

        self.weights.index = assets[self.ordered_indices]
        self.weights = pd.DataFrame(self.weights)
        self.weights = self.weights.T

    def plot_clusters(self, assets):
        dendrogram_plot = dendrogram(self.clusters, labels=assets)
        return dendrogram_plot

    @staticmethod
    def _shrink_covariance(covariance):
        oas = OAS()
        oas.fit(covariance)
        shrinked_covariance = oas.covariance_
        return pd.DataFrame(shrinked_covariance, index=covariance.columns, columns=covariance.columns)

    @staticmethod
    def _cov2corr(covariance):
        d_matrix = np.zeros_like(covariance)
        diagnoal_sqrt = np.sqrt(np.diag(covariance))
        np.fill_diagonal(d_matrix, diagnoal_sqrt)
        d_inv = np.linalg.inv(d_matrix)
        corr = np.dot(np.dot(d_inv, covariance), d_inv)
        corr = pd.DataFrame(corr, index=covariance.columns, columns=covariance.columns)
        return corr

    def allocate(self,
                 asset_names,
                 asset_prices=None,
                 asset_returns=None,
                 covariance_matrix=None,
                 resample_by=None,
                 use_shrinkage=False):
        if asset_prices is None and asset_returns is None and covariance_matrix is None:
            raise ValueError("You need to supply either raw prices or returns or a covariance matrix of asset returns")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        if asset_returns is None and covariance_matrix is None:
            asset_returns = self.returns_estimator.calculate_returns(asset_prices=asset_prices, resample_by=resample_by)
        asset_returns = pd.DataFrame(asset_returns, columns=asset_names)

        if covariance_matrix is None:
            covariance_matrix = asset_returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        if use_shrinkage:
            cov = self._shrink_covariance(covariance=cov)

        corr = self._cov2corr(covariance=cov)

        distances, self.clusters = self._tree_clustering(correlation=corr)

        num_assets = len(asset_names)
        self.ordered_indices = self._quasi_diagnalization(num_assets, 2 * num_assets - 2)
        self.seriated_distances, self.seriated_correlations = self._get_seriated_matrix(assets=asset_names,
                                                                                        distances=distances,
                                                                                        correlations=corr)


        self._recursive_bisection(covariance=cov, assets=asset_names)