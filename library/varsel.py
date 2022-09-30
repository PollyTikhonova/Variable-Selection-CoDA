from skbio.stats.composition import clr, ilr, multiplicative_replacement
import scipy.stats as st
import numpy as np
from warnings import warn

def normalize(matrix):
    '''
    matrix: np.matrix 
    (rows - elements, columns - features)
    '''
    return matrix / matrix.sum(axis=1).reshape(-1, 1)

def Hron_Kubacek_test(a, b, matrix):
    '''
    a: float, totvar of this step;
    b: float, totvar of previous step
    matrix: subcomposition matrix of this step

    return: statistic, pvalue
    '''
    ilr_matrix = ilr(multiplicative_replacement(matrix))
    n = ilr_matrix.shape[0]
    covariance = np.cov(ilr_matrix.transpose())
    statistic = n/(n-1)*(a - b) / np.sqrt(
        2 / (n - 1) * np.trace(covariance.dot(covariance)))
    return statistic, st.norm.cdf(statistic)

def totvar(variances):
    return np.sum(variances, axis=0)


def feature_selection(df,
                      alpha=.05,
                      return_df=True,
                      return_column_names=False,
                      to_the_end=True, 
                      return_variances=False, 
                      return_columns_que=False):
    '''
    df: pandas.DataFrame
    alpha: significance level for Hron and Kubácek test
    return: df with selected columns and/or column names
    '''

    def selection_step(matrix, columns):
        matrix = normalize(matrix)
        clr_matrix = clr(multiplicative_replacement(matrix))
        variances = np.var(clr_matrix, axis=0)
        col_index = list(range(matrix.shape[1]))
        col_index.remove(np.argmin(variances))
        return totvar(variances), matrix[:, col_index], columns[col_index]

    columns_que = []
    total_variances = []
    significant_step = None

    matrix = df.values
    columns = [np.array(df.columns)]
    totvar_, matrix_, columns_ = selection_step(matrix, columns[0])
    total_variances.append(totvar_)
    columns_que.append(list(set(columns[0]) - set(columns_))[0])
    while (len(columns_) > 2):
        matrix = matrix_
        columns.append(columns_)
        totvar_, matrix_, columns_ = selection_step(matrix, columns_)
        total_variances.append(totvar_)
        statistics, pvalue = Hron_Kubacek_test(total_variances[-1], total_variances[-2], matrix)
        n = matrix.shape[0]
        if (pvalue <= alpha):
            if significant_step is None:
                significant_step = len(df.columns) - len(columns[0])
            if not to_the_end:
                columns_que.pop()
                break
        columns_que.append(list(set(columns[-1]) - set(columns_))[0])
        columns.pop(0)
    if significant_step is None:
           warn('The covariance did not changed significantly! So, all features are gone...')
    columns = columns[0] if not to_the_end else columns_
    to_return = []
    if return_df:
        to_return.append(df[columns])
    if return_column_names:
        to_return.append(columns)
    if to_the_end:
        to_return.append(significant_step)    
    if return_variances:
        to_return.append(total_variances)
    if return_columns_que:
        to_return.append(columns_que)
    return to_return