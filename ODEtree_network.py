import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
import time
from multiprocessing import Pool


def ODEtree_network(TS_data, time_points, alpha='from_data', SS_data=None, gene_names=None, regulators='all',
               tree_method='RF', tree_kwargs=None, remove_output=False, nthreads=1):
    '''Computation of tree-based scores for all putative regulatory links.

    Parameters
    ----------

    TS_data: list of numpy arrays
        List of arrays, where each array contains the gene expression values of one time series experiment. Each row of an array corresponds to a time point and each column corresponds to a gene. The i-th column of each array must correspond to the same gene.

    time_points: list of one-dimensional numpy arrays
        List of n vectors, where n is the number of time series (i.e. the number of arrays in TS_data), containing the time points of the different time series. The i-th vector specifies the time points of the i-th time series of TS_data.

    alpha: either 'from_data', a positive number or a vector of positive numbers
        Specifies the degradation rate of the different gene expressions.
        When alpha = 'from_data', the degradation rate of each gene is estimated from the data, by assuming an exponential decay between the highest and lowest observed expression values.
        When alpha is a vector of positive numbers, the i-th element of the vector must specify the degradation rate of the i-th gene.
        When alpha is a positive number, all the genes are assumed to have the same degradation rate alpha.
        default: 'from_data'

    SS_data: numpy array, optional
        Array containing steady-state gene expression values. Each row corresponds to a steady-state condition and each column corresponds to a gene. The i-th column/gene must correspond to the i-th column/gene of each array of TS_data.
        default: None

    gene_names: list of strings, optional
        List of length p containing the names of the genes, where p is the number of columns/genes in each array of TS_data. The i-th item of gene_names must correspond to the i-th column of each array of TS_data (and the i-th column of SS_data when SS_data is not None).
        default: None

    regulators: list of strings, optional
        List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names). When regulators is set to 'all', any gene can be a candidate regulator.
        default: 'all'

    tree_method: 'RF' or 'XGB', optional
        Specifies which tree-based procedure is used: either Random Forest ('RF') or XGBoost ('XGB').
        default: 'RF'

    tree_kwargs: dictionary comprising the parameters of the tree-based method, optional
        default: dict(n_estimators=1000)

    remove_output: boolean, optional
        Indicates whether or not to remove the target gene j from the candidate regulators when learning model f_j.
        default: False

    nthreads: positive integer, optional
        Number of threads used for parallel computing
        default: 1


    Returns
    -------

    VIM: a dictionary in which VIM[importance_type] is an array where the element (i,j) is the score of the edge directed from the i-th gene to the j-th gene.
    All diagonal elements are set to zero (auto-regulations are not considered).
    When a list of candidate regulators is provided, all the edges directed from a gene that is not a candidate
    regulator are set to zero.
    For RF, importance_type is 'MDI'.
    For XGB, importance_type can be 'weight', 'gain', 'total_gain', 'cover', or 'total_cover'?


    '''

    time_start = time.time()

    # Check input arguments
    if not isinstance(TS_data, (list, tuple)):
        raise ValueError(
            'TS_data must be a list of arrays, where each row of an array corresponds to a time point/sample and each'
            'column corresponds to a gene')

    for expr_data in TS_data:
        if not isinstance(expr_data, np.ndarray):
            raise ValueError(
                'TS_data must be a list of arrays, where each row of an array corresponds to a time point/sample'
                'and each column corresponds to a gene')

    ngenes = TS_data[0].shape[1]

    if len(TS_data) > 1:
        for expr_data in TS_data[1:]:
            if expr_data.shape[1] != ngenes:
                raise ValueError('The number of columns/genes must be the same in every array of TS_data.')

    if not isinstance(time_points, (list, tuple)):
        raise ValueError(
            'time_points must be a list of n one-dimensional arrays, where n is the number of time series experiments'
            'in TS_data')

    if len(time_points) != len(TS_data):
        raise ValueError(
            'time_points must be a list of n one-dimensional arrays, where n is the number of time series experiments'
            'in TS_data')

    for tp in time_points:
        if (not isinstance(tp, (list, tuple, np.ndarray))) or (isinstance(tp, np.ndarray) and tp.ndim > 1):
            raise ValueError(
                'time_points must be a list of n one-dimensional arrays, where n is the number of time series in'
                'TS_data')

    for (i, expr_data) in enumerate(TS_data):
        if len(time_points[i]) != expr_data.shape[0]:
            raise ValueError(
                'The length of the i-th vector of time_points must be equal to the number of rows in the i-th array of'
                'TS_data')

    if alpha != 'from_data':
        if not isinstance(alpha, (list, tuple, np.ndarray, int, float)):
            raise ValueError(
                "input argument alpha must be either 'from_data', a positive number or a vector of positive numbers")

        if isinstance(alpha, (int, float)) and alpha < 0:
            raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")

        if isinstance(alpha, (list, tuple, np.ndarray)):
            if isinstance(alpha, np.ndarray) and alpha.ndim > 1:
                raise ValueError(
                    "input argument alpha must be either 'from_data', a positive number or a vector of positive numbers")
            if len(alpha) != ngenes:
                raise ValueError(
                    'when input argument alpha is a vector, this must be a vector of length p, where p is the number of genes')
            for a in alpha:
                if a < 0:
                    raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")

    if SS_data is not None:
        if not isinstance(SS_data, np.ndarray):
            raise ValueError(
                'SS_data must be an array in which each row corresponds to a steady-state condition/sample and each'
                'column corresponds to a gene')

        if SS_data.ndim != 2:
            raise ValueError(
                'SS_data must be an array in which each row corresponds to a steady-state condition/sample and each'
                'column corresponds to a gene')

        if SS_data.shape[1] != ngenes:
            raise ValueError(
                'The number of columns/genes in SS_data must by the same as the number of columns/genes in every array'
                'of TS_data.')

    if gene_names is not None:
        if not isinstance(gene_names, (list, tuple)):
            raise ValueError('input argument gene_names must be a list of gene names')
        elif len(gene_names) != ngenes:
            raise ValueError(
                'input argument gene_names must be a list of length p, where p is the number of columns/genes in the'
                'expression data')

    if regulators != 'all':
        if not isinstance(regulators, (list, tuple)):
            raise ValueError('input argument regulators must be a list of gene names')

        if gene_names is None:
            raise ValueError('the gene names must be specified (in input argument gene_names)')
        else:
            sIntersection = set(gene_names).intersection(set(regulators))
            if not sIntersection:
                raise ValueError('The genes must contain at least one candidate regulator')

    if tree_method not in ['RF', 'XGB']:
        raise ValueError('input argument tree_method must be "RF" (Random Forests) or "XGB" (XGBoost)')

    if not isinstance(nthreads, int):
        raise ValueError('input argument nthreads must be a stricly positive integer')
    elif nthreads <= 0:
        raise ValueError('input argument nthreads must be a stricly positive integer')

    # Re-order time points in increasing order
    for (i, tp) in enumerate(time_points):
        tp = np.array(tp, np.float32)
        indices = np.argsort(tp)
        time_points[i] = tp[indices]
        expr_data = TS_data[i]
        TS_data[i] = expr_data[indices, :]

    # Decay rates
    if alpha == 'from_data':
        alphas = estimate_degradation_rates(TS_data, time_points)
    elif isinstance(alpha, (int, float)):
        alphas = np.zeros(ngenes) + float(alpha)
    else:
        alphas = [float(a) for a in alpha]

    print('Tree method: ' + str(tree_method))
    print('alpha min: ' + str(min(alphas)))
    print('alpha max: ' + str(max(alphas)))
    print('\n')

    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = list(range(ngenes))
    else:
        input_idx = [gene_names.index(regulator) for regulator in regulators]

    nregulators = len(input_idx)

    # Importance types to compute
    if tree_method == 'RF' or tree_method == 'ET':
        importance_types = ['MDI']
    else:
        importance_types = ['weight', 'gain', 'total_gain', 'cover', 'total_cover']

    # Hyper-parameters of the tree-based method
    if tree_kwargs is None:
        tree_kwargs = dict()

    # Learn an ensemble of trees for each target gene and compute scores for candidate regulators
    VIM = dict()
    for importance_type in importance_types:
        VIM[importance_type] = np.zeros((nregulators, ngenes))

    if nthreads > 1:

        print('running jobs on %d threads' % nthreads)

        input_data = [[TS_data, time_points, SS_data, i, alphas[i], input_idx,
                       tree_method, tree_kwargs, importance_types, remove_output] for i in range(ngenes)]

        pool = Pool(nthreads)
        alloutput = pool.map(wr_GENIE3_ODE_single, input_data)

        for (i, vi) in alloutput:

            for importance_type in importance_types:
                VIM[importance_type][:, i] = vi[importance_type]

    else:

        print('running single threaded jobs')
        for i in range(ngenes):

            print('Gene %d/%d...' % (i + 1, ngenes))

            vi = GENIE3_ODE_single(TS_data, time_points, SS_data, i, alphas[i], input_idx,
                                   tree_method, tree_kwargs, importance_types, remove_output)

            for importance_type in importance_types:
                VIM[importance_type][:, i] = vi[importance_type]

    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM


def wr_GENIE3_ODE_single(args):
    return ([args[3], GENIE3_ODE_single(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
                                        args[8], args[9])])


def GENIE3_ODE_single(TS_data, time_points, SS_data, output_idx, alpha, input_idx,
                      tree_method, tree_kwargs, importance_types, remove_output):
    # lag (in number of time points) used for the finite approximation of the derivative of the target gene expression
    h = 1

    input_idx_final = input_idx[:]
    if remove_output and output_idx in input_idx:
        input_idx_final.remove(output_idx)

    nexp = len(TS_data)
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
    ninputs = len(input_idx_final)

    # Construct learning sample

    # Time-series data
    input_matrix_time = np.zeros((nsamples_time - h * nexp, ninputs))
    output_vect_time = np.zeros(nsamples_time - h * nexp)

    nsamples_count = 0

    for (i, current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        npoints = current_timeseries.shape[0]
        current_timeseries_input = current_timeseries[:npoints - h, input_idx_final]
        time_diff_current = current_time_points[h:] - current_time_points[:npoints - h]
        output_diff_current = current_timeseries[h:, output_idx] - current_timeseries[:npoints - h, output_idx]
        current_timeseries_output = output_diff_current / time_diff_current + alpha * current_timeseries[:npoints - h,
                                                                                      output_idx]
        nsamples_current = current_timeseries_input.shape[0]
        input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current

    # Steady-state data (if any)
    if SS_data is not None:

        input_matrix_steady = SS_data[:, input_idx_final]
        output_vect_steady = SS_data[:, output_idx] * alpha

        # Concatenation
        input_all = np.vstack([input_matrix_steady, input_matrix_time])
        output_all = np.concatenate((output_vect_steady, output_vect_time))

    else:

        input_all = input_matrix_time
        output_all = output_vect_time

    # Tree-based method
    if tree_method == 'RF':
        treeEstimator = RandomForestRegressor(**tree_kwargs)
    else:
        treeEstimator = XGBRegressor(**tree_kwargs)

    # Learn ensemble of trees
    treeEstimator.fit(input_all, output_all)

    # Compute importance scores
    importances = compute_feature_importances(treeEstimator, importance_types)

    vi = dict()
    ngenes = TS_data[0].shape[1]

    for importance_type in importances:
        vi_current = np.zeros(ngenes)
        vi_current[input_idx_final] = importances[importance_type]

        # Remove self-regulation
        vi_current[output_idx] = 0

        # Normalize importance scores
        importances_sum = sum(vi_current)
        if importances_sum > 0:
            vi_current = vi_current / importances_sum

        vi[importance_type] = vi_current[input_idx]

    return vi


def compute_feature_importances(estimator, importance_types):
    """Computes variable importances from a trained tree-based model.
    """

    importances = dict()

    if isinstance(estimator, XGBRegressor):

        nfeatures = estimator.n_features_in_
        features = ['f%d' % i for i in range(nfeatures)]

        for importance_type in importance_types:
            dict_imp = estimator.get_booster().get_score(importance_type=importance_type)
            importances[importance_type] = np.array([dict_imp[feature] if feature in dict_imp.keys() else 0
                                                     for feature in features])

    else:
        for importance_type in importance_types:

            if importance_type == 'MDI':
                importances_mdi = [e.tree_.compute_feature_importances(normalize=False) for e in estimator.estimators_]
                importances[importance_type] = np.mean(np.array(importances_mdi), axis=0)

            else:
                raise ValueError('Unknown importance type for RF: %s' % importance_type)

    return importances


def estimate_degradation_rates(TS_data, time_points):
    """
    For each gene, the degradation rate is estimated by assuming that the gene expression x(t) follows:
    x(t) =  A exp(-alpha * t) + C_min,
    between the highest and lowest expression values.
    C_min is set to the minimum expression value over all genes and all samples.
    """

    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)

    C_min = TS_data[0].min()
    if nexp > 1:
        for current_timeseries in TS_data[1:]:
            C_min = min(C_min, current_timeseries.min())

    alphas = np.zeros((nexp, ngenes))

    for (i, current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]

        for j in range(ngenes):
            idx_min = np.argmin(current_timeseries[:, j])
            idx_max = np.argmax(current_timeseries[:, j])

            xmin = current_timeseries[idx_min, j]
            xmax = current_timeseries[idx_max, j]

            tmin = current_time_points[idx_min]
            tmax = current_time_points[idx_max]

            xmin = max(xmin - C_min, 1e-6)
            xmax = max(xmax - C_min, 1e-6)

            xmin = np.log(xmin)
            xmax = np.log(xmax)

            alphas[i, j] = (xmax - xmin) / abs(tmin - tmax)

    alphas = np.amax(alphas, axis=0)

    return alphas


def get_gold_standard(filename, gene_names, regulators, only_present_edges=False):
    ngenes = len(gene_names)

    if regulators == 'all':
        input_idx = list(range(ngenes))
    else:
        input_idx = [gene_names.index(regulator) for regulator in regulators]

    d_indices = dict()
    for i, gene_name in enumerate(gene_names):
        d_indices[gene_name] = i

    gold = np.zeros((ngenes, ngenes))

    with open(filename) as f:

        if only_present_edges:

            for line in f:
                regulator, target = line.rstrip('\n').split('\t')
                i = d_indices[regulator]
                j = d_indices[target]
                gold[i, j] = 1

        else:
            for line in f:
                regulator, target, score = line.rstrip('\n').split('\t')
                i = d_indices[regulator]
                j = d_indices[target]
                gold[i, j] = int(score)

    return gold[input_idx, :]


def scores(VIM, gold):
    pred = VIM.flatten()
    true = gold.flatten()

    aupr = average_precision_score(true, pred)
    auroc = roc_auc_score(true, pred)

    return auroc, aupr
