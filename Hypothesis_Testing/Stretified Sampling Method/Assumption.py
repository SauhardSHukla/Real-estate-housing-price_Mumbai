#Import the Library

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

    # --- Visualization ---(if want all the group in one graph)
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
        _, axes = plt.subplots(nrows=n, figsize=(10, 4 * n))
    
        for ax, grp in zip(axes, grp_vals):
            sns.histplot(df[df[grp_col] == grp][val_col], ax=ax, kde=True, bins=30, color='skyblue')
            ax.set_title(f'{val_col} Distribution for {grp_col} = {grp}')
            ax.set_xlabel(val_col)
            ax.set_ylabel('Density')
    
        plt.tight_layout()
        plt.show()

    return summary
