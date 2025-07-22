import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, chisquare


def visualize_chi_square_contingency(df, row_var, col_var):
    """
    Visualizes a contingency table with a heatmap and performs chi-square test of independence.
    """
    contingency = pd.crosstab(df[row_var], df[col_var])
    
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)

    # Plot heatmap
    print(f'Graph for the Visualizes a contingency table  ')
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency, annot=True, cmap="YlGnBu", fmt="d")
    plt.title(f"Contingency Table: {row_var} vs {col_var}\nChi2: {chi2:.2f}, p-value: {p:.4f}")
    plt.ylabel(row_var)
    plt.xlabel(col_var)
    plt.tight_layout()
    plt.show()
    

    return chi2, p


def visualize_chi_square_goodness_of_fit(series):
    """
    Visualizes observed vs expected frequencies for a single categorical variable.
    Assumes uniform expected distribution.
    """
    observed = series.value_counts().sort_index()
    expected = [observed.sum() / len(observed)] * len(observed)

    # Chi-square test
    chi2_stat, p = chisquare(f_obs=observed, f_exp=expected)

    # Bar plot: side-by-side
    x = np.arange(len(observed.index))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, observed.values, width=0.4, label='Observed')
    plt.bar(x + 0.2, expected, width=0.4, label='Expected')
    plt.xticks(x, observed.index, rotation=45)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title(f"Observed vs Expected Frequencies\nChi2: {chi2_stat:.2f}, p-value: {p:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return chi2_stat, p
