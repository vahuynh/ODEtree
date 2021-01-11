# Hybrid ODE/tree ensemble approach for the inference of gene regulatory networks.

The method is implemented in Python and allow to infer gene networks from time series data alone, or jointly from time series and steady-state data.

## Usage

```
from ODEtree_network import ODEtree_network

VIM = ODEtree_network(TS_data, time_points, alpha='from_data', SS_data=None, gene_names=None, regulators='all', tree_method='RF', tree_kwargs=None, remove_output=False, nthreads=1)
```
- `TS_data`: List of arrays, where each array contains the gene expression values of one time series experiment. Each row of an array corresponds to a time point and each column corresponds to a gene. The i-th column of each array must correspond to the same gene.
- `time_points`: List of *n* vectors, where *n* is the number of time series (i.e. the number of arrays in `TS_data`), containing the time points of the different time series. The i-th vector specifies the time points of the i-th time series of `TS_data`.
- `alpha`: Specifies the degradation rate of the different gene expressions. When `alpha` = 'from_data', the degradation rate of each gene is estimated from the data, by assuming an exponential decay between the highest and lowest observed expression values. When `alpha` is a vector of positive numbers, the i-th element of the vector must specify the degradation rate of the i-th gene. When `alpha` is a positive number, all the genes are assumed to have the same degradation rate alpha.
