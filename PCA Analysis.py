import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pandas as pd
dataset_path = r"C:\Users\harsh\Desktop\dv\Cleaned_Dataset.xlsx"
df = pd.read_excel(dataset_path)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.show()

import numpy as np
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Optimal number of components: {optimal_components}")

from sklearn.decomposition import PCA
pca_optimal = PCA(n_components=optimal_components)
pca_transformed = pca_optimal.fit_transform(df_scaled)

pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(optimal_components)])
print(pca_df.head())
