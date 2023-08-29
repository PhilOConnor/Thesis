import os
import pandas as pd
import numpy as np

uni_data_path = '../Outputs/csv/uniform dist/'
class_data_path = '../Outputs/csv/class weights/'

uni_files = os.listdir(uni_data_path)
class_files = os.listdir(class_data_path)
d = np.array([])
for file in uni_files:
	df = pd.read_csv(os.path.join(uni_class_path ,file), skip_rows=1, n_rows=1)
	d = np.concat(d, df)
df = pd.DataFrame(d)
print(d)
