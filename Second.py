from utilities import *

# Load all sheets from the Excel file
file_path='/Users/habeeb/Downloads/Git/ML/Petroleum-Target-Xplorer/Data/Excel Sheets/Passey.xlsx'
dfs=load_excel_sheets(file_path)

# Create list of tuples containing DataFrame and sheet name
df_list = [(df, sheet_name) for sheet_name, df in dfs.items()]

# Access the loaded DataFrames
data_summary_df = dfs['Data Summary']
KR2 = dfs['TOC-ESTM (KR-2)']
KR3_edit = dfs['TOC-ESTM (KR-3) (edittt)']
KR3=dfs['TOC-ESTM (KR-3)']
sheet2_df = dfs['Sheet2']
sheet1_df = dfs['Sheet1']

# Drop 'MD' column from KR2 dataframe
KR2 = KR2.drop('MD', axis=1)

# Drop 'MD'
#measure Toc and Measured-TOC.1 in KR3 are same, so we drop one
# Drop 'Measured-TOC.1' column from KR3 dataframe, likwise TOC RK eval
KR3 = KR3.drop(['MD','Measured-TOC.1','TOC (Rk-Eval)'], axis=1)

#This is because in the MVA, there was no signle equation that considered MD, TOC (Rk-Eval)
# Drop unnamed columns from the DataFrame
KR2 = drop_unnamed_columns(KR2)
KR3 = drop_unnamed_columns(KR3)

print('KR2 Well log descriptive statistics')
print(KR2.describe().T)

print('KR3 Well log descriptive statistics')
print(KR3.describe().T)



