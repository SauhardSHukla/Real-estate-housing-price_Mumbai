# # Import of the Library

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import normaltest, levene

# def dignostics(df,grp_col='bhk',val_col='rnet',vis=True):
#   """
#     Performs normality test for each group, Levene test for equal variances,
#     and optionally plots distributions.

#     Parameters:
#         df (pd.DataFrame): Input data.
#         group_col (str): Categorical column to group by (e.g., 'bhk').
#         value_col (str): Numerical column to test (e.g., ' rent_monthly').
#         alpha (float): Significance level for tests.
#         visualize (bool): If True, plots group-wise distributions.

#     Returns:
#         dict: Summary of normality and Levene test results.
#     """
  
#   summary ={}
#   grp_cols = df[grp_col].dropna().unique()
#   normality_test ={}

#   # --- Normality Test Per Group ---
#   for grp in grp_cols:
#     subset = df[df[grp_col]==grp][val_col].dropna()

#     if len(subset)>=8:
#       stats,p = normaltest(subset)
#       result = 'Normal' if p>0.05 else 'Not Normal'
#       normality_test[grp] = {'statistic:': stats, 'p-value ': p, 'result ': result}
#     else:
#       normality_test[grp] = {'statistic': None, 'p_value': None, 'result': 'Too few samples'}
  
#   summary['normality']= normality_test

#   # --- Levene’s Test for Homogeneity of Variance ---
#   group_values = [df[df[grp_col]==grp][val_col].dropna() for grp in grp_cols if len(df[df[grp_col]==grp])>=2]
#   if len(group_values) >= 2:
#         levene_stat, levene_p = levene(*group_values)
#         levene_result = 'Equal Variances' if levene_p > 0.05 else 'Unequal Variances'
#   else:
#         levene_stat, levene_p, levene_result = None, None, 'Insufficient groups'

#   summary['levene'] = {
#         'statistic': levene_stat,
#         'p_value': levene_p,
#         'result': levene_result
#     }
  
#   if vis:
#      plt.figure(figsize=(10,12))
#      sns.hist(data=df,hue=grp_cols,x=val_col,kde=True,bins=30,element='step',stat ='density',common_norm =False)
#      plt.title(f'Distribution of {val_col} grouped by {grp_col}')
#      plt.xlabel(val_col)
#      plt.ylabel('Density')
#      plt.tight_layout()
#      plt.show()

#   return summary
  

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest, levene

def diagnostics(df, grp_col='bhk', val_col='rnet', vis=True, alpha=0.05):
    """
    Performs normality test (D'Agostino's K²) for each group, Levene's test for homogeneity of variances,
    and optionally plots the distributions grouped by a categorical column.

    Parameters:
        df (pd.DataFrame): Input dataset.
        grp_col (str): Column to group by (categorical).
        val_col (str): Column to test (numerical).
        vis (bool): If True, generates histogram plots per group.
        alpha (float): Significance level for statistical tests.

    Returns:
        dict: Summary of normality test and Levene test results.
    """

    summary = {}
    normality_test = {}

    # Get unique non-null group values
    group_names = df[grp_col].dropna().unique()

    # --- Normality Test (per group) ---
    for group in group_names:
        subset = df[df[grp_col] == group][val_col].dropna()

        if len(subset) >= 8:
            stat, p_val = normaltest(subset)
            normality_test[group] = {
                'statistic': stat,
                'p_value': p_val,
                'result': 'Normal' if p_val > alpha else 'Not Normal'
            }
        else:
            normality_test[group] = {
                'statistic': None,
                'p_value': None,
                'result': 'Too few samples (<8)'
            }

    summary['normality'] = normality_test

    # --- Levene’s Test (for homogeneity of variances) ---
    grouped_vals = [
        df[df[grp_col] == group][val_col].dropna()
        for group in group_names
        if len(df[df[grp_col] == group][val_col].dropna()) >= 2
    ]

    if len(grouped_vals) >= 2:
        levene_stat, levene_p = levene(*grouped_vals)
        summary['levene'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'result': 'Equal Variances' if levene_p > alpha else 'Unequal Variances'
        }
    else:
        summary['levene'] = {
            'statistic': None,
            'p_value': None,
            'result': 'Insufficient data across groups'
        }

    # --- Visualization ---
    # if vis:
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(data=df, x=val_col, hue=grp, kde=True,
    #                  bins=30, element='step', stat='density', common_norm=False)
    #     plt.title(f'Distribution of {val_col} by {grp_col}')
    #     plt.xlabel(val_col)
    #     plt.ylabel('Density')
    #     plt.tight_layout()
    #     plt.show()
    
    if vis:
        grp_vals = df[grp_col].dropna().unique()
        n = len(grp_vals)
        fig, axes = plt.subplots(nrows=n, figsize=(10, 4 * n))
    
        for ax, grp in zip(axes, grp_vals):
            sns.histplot(df[df[grp_col] == grp][val_col], ax=ax, kde=True, bins=30, color='skyblue')
            ax.set_title(f'{val_col} Distribution for {grp_col} = {grp}')
            ax.set_xlabel(val_col)
            ax.set_ylabel('Density')
    
        plt.tight_layout()
        plt.show()



    return summary
