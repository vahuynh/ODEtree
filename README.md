# Hybrid ODE/tree ensemble approach for the inference of gene regulatory networks.

The method is implemented in Python and allow to infer gene networks from time series data alone, or jointly from time series and steady-state data.

## Usage

```
from ODEtree_network import ODEtree_network

VIM = ODEtree_network(TS_data, time_points, alpha='from_data', SS_data=None, gene_names=None, regulators='all', tree_method='RF', tree_kwargs=None, remove_output=False, nthreads=1)
```
- `TS_data`: List of arrays, where each array contains the gene expression values of one time series experiment. Each row of an array corresponds to a time point and each column corresponds to a gene. The i-th column of each array must correspond to the same gene.
- `time_points`: List of *n* vectors, where *n* is the number of time series (i.e. the number of arrays in `TS_data`), containing the time points of the different time series. The *i*-th vector specifies the time points of the *i*-th time series of `TS_data`.
- `alpha`: Specifies the degradation rate of the different gene expressions. When `alpha` = 'from_data', the degradation rate of each gene is estimated from the data, by assuming an exponential decay between the highest and lowest observed expression values. When `alpha` is a vector of positive numbers, the *i*-th element of the vector must specify the degradation rate of the *i*-th gene. When `alpha` is a positive number, all the genes are assumed to have the same degradation rate alpha.
- `SS_data`: (optional) Array containing steady-state gene expression values. Each row corresponds to a steady-state condition and each column corresponds to a gene. The *i*-th column/gene must correspond to the *i*-th column/gene of each array of `TS_data`.
- `gene_names`: (optional) List of length *p* containing the names of the genes, where *p* is the number of columns/genes in each array of `TS_data`. The *i*-th item of `gene_names` must correspond to the *i*-th column of each array of `TS_data` (and the *i*-th column of `SS_data` when `SS_data` is not None).
- `regulators`: List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in `gene_names`). When `regulators` is set to 'all', any gene can be a candidate regulator.
- `tree_method`: Either 'RF' or 'XGB'. Specifies which tree-based procedure is used: either Random Forest ('RF') or XGBoost ('XGB').
- `tree_kwargs`: Dictionary comprising the hyper-parameters of the tree-based method. The hyper-parameters and the ones of the scikit-learn [RandomForestRegressor class](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) for RF and of the XGBoost [XGBRegresor class](https://xgboost.readthedocs.io/en/latest/python/python_api.html) for XGB. *Note*: all the results in the paper were obtained with `random_state`=0.
- 'remove_output': Boolean indicating whether or not to remove the target gene *j* from the candidate regulators when learning model *f_j*.
- 'nthreads': Number of threads used for parallel computing. 

Returns: a dictionary `VIM`, in which `VIM[importance_type]` is an array where the element (*i,j*) is the score of the edge directed from the *i*-th regulator to the *j*-th gene. 
Auto-regulations, i.e. links from a regulator to itself, are set to zero.
For RF, `importance_type` is 'MDI'.
For XGB, `importance_type` can be 'weight', 'gain', 'total_gain', 'cover', or 'total_cover'.

---

## Example

The example below shows how to infer a network from the Drosophila dataset, and how to benchmark it against the gold standard network. We use here the Random forest method with 100 trees.

```
from ODEtreenetwork import ODEtreenetwork, get_gold_standard, scores
import _pickle

#Load data
with open('datasets/drosophila/drosophila_data.pkl', 'rb') as f:
	(TS_data, time_points, gene_names, regulators, alphas) = _pickle.load(f)

#Infer network
VIM = ODEtree_network(TS_data, time_points, alpha=alphas, gene_names=gene_names, regulators=regulators, tree_method='RF', tree_kwargs=dict(n_estimators=100))

#Load gold standard
goldfile = 'datasets/drosophila/drosophila_gold_standard.txt'
goldnet = get_gold_standard(goldfile, gene_names, regulators, only_present_edges=True)

#Compute areas under ROC curve and precision-recall curve
auroc, aupr = scores(VIM['MDI'], goldnet)
```
