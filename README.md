# Hybrid ODE/tree ensemble approach for the inference of gene regulatory networks.

The method is implemented in Python and allow to infer gene networks from time series data alone, or jointly from time series and steady-state data.

## Usage

```
from ODEtree_network import ODEtree_network

VIM = ODEtree_network(TS_data, time_points, alpha='from_data', SS_data=None, gene_names=None, regulators='all', tree_method='RF', tree_kwargs=None, remove_output=False, nthreads=1)
```
- `TS_data`: List of arrays, where each array contains the gene expression values of one time series experiment. Each row of an array corresponds to a time point and each column corresponds to a gene. The i-th column of each array must correspond to the same gene.
