import pandas as pd
import os

def merge_csv(input_files, output_file):
    # 读取并合并所有文件
    df_list = []
    for file in input_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    # 合并所有的DataFrame
    df_merged = pd.concat(df_list, ignore_index=True)
    
    # 保存合并后的文件
    df_merged.to_csv(output_file, index=False)
    print(f"文件已合并并保存为：{output_file}")

# 使用方法
input_files = ['output_file_part_1.csv', 'output_file_part_2.csv', 'output_file_part_3.csv', 'output_file_part_4.csv', 'output_file_part_5.csv']  # 列出所有要合并的文件
merge_csv(input_files, 'data_del.csv')
