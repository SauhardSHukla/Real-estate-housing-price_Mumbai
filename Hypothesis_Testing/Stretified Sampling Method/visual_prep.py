"""

Here we will create a function that will help us to show the visual format
of the ANOVA or any other method visual form

"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def Visual_format(df,grp_col,value_col,group):
      
      #Get the Filtered Data 
      filtered_df = df[df[grp_col].isin(group)]

      # Boxplot to show distributions
      plt.figure(figsize=(12, 6))
      sns.boxplot(x=grp_col, y=value_col, data=filtered_df)
      plt.title(f"Boxplot of {value_col} for {', '.join(group)}")
      plt.xticks(rotation=45)
      plt.grid(True)
      plt.tight_layout()
      plt.show()

      # Pointplot to show group means with confidence intervals
      plt.figure(figsize=(12, 6))
      sns.pointplot(x=grp_col, y=value_col, data=filtered_df, ci="sd", join=False, capsize=.2)
      plt.title(f"Mean Plot of {value_col}  for {', '.join(group)}")
      plt.xticks(rotation=45)
      plt.grid(True)
      plt.tight_layout()
      plt.show()