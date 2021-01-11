# Hybrid ODE/tree ensemble approach for the inference of gene regulatory networks.

The method is implemented in Python.

## Usage

```
from ODEtree_network import ODEtree_network

VIM = ODEtree_network(TS_data, time_points, alpha='from_data', SS_data=None, gene_names=None, regulators='all',
               tree_method='RF', tree_kwargs=None, remove_output=False, nthreads=1)
```