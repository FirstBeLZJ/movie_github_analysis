import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
import pandas as pd
from tqdm import tqdm

github_df = pd.read_csv('github/repository_data.csv')

rows_with_missing = github_df[github_df.isnull().any(axis=1)]
numeric_rows_with = rows_with_missing[['stars_count','forks_count','watchers','pull_requests','commit_count']]
# print(numeric_rows_with)
rows_without_missing = github_df[~github_df.isnull().any(axis=1)]
#'stars_count','forks_count','issues_count','pull_requests','contributors'
numeric_rows_without = rows_without_missing[['stars_count','forks_count','watchers','pull_requests','commit_count']]
# print(numeric_rows_without)
interpoted = rows_with_missing.copy()
for i, row in tqdm(enumerate(numeric_rows_with.itertuples()), total=numeric_rows_with.shape[0]):
    # 计算当前行与其他行的欧氏距离，忽略NaN值
    distances = nan_euclidean_distances([numeric_rows_with.loc[row.Index]], numeric_rows_without)[0]
    nearest_row_index = np.argmin(distances)
    nearest_row = rows_without_missing.iloc[nearest_row_index]
    # 使用最近邻行的值填充缺失值
    # print(row.Index)
    # print("000",row)
    # print("111",interpoted.iloc[i])
    # print("222",nearest_row)
    interpoted.loc[row.Index] = nearest_row

print(interpoted)