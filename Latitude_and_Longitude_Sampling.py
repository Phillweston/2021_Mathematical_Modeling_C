# This is a latitude and longitude transform script
import pandas as pd
import numpy as np
import math
from pandas.core.frame import DataFrame


# 读取2019年的经纬度以及Lab Status数据
Latitude_and_Longitude_Data = pd.read_excel('Proceed_Data2019.xlsx')
# 设置经纬度最大最小值
Latitude_max = Latitude_and_Longitude_Data['Latitude'].max()
Latitude_min = Latitude_and_Longitude_Data['Latitude'].min()
Longitude_max = Latitude_and_Longitude_Data['Longitude'].max()
Longitude_min = Latitude_and_Longitude_Data['Longitude'].min()
# print(Latitude_max, Latitude_min)
# print(Latitude_and_Longitude_Data.shape[0])
# 定义经纬度比例尺（即一度代表多少km）
Latitude_scale = 111
Longitude_scale = 111 * math.cos(Latitude_scale)
# 用于存放经纬度转化之后的数据以及对应的Positive判断值
# Latitude_and_Longitude_Array = np.zeros([Latitude_and_Longitude_Data.shape[0], 3])
# 定义绘图坐标系的范围
Plot_scale = 2
Plot_x_range = abs(round((Latitude_max - Latitude_min) * Latitude_scale / Plot_scale))
Plot_y_range = abs(round((Longitude_max - Longitude_min) * Longitude_scale / Plot_scale))
print(Plot_x_range, Plot_y_range)
# 用于存放经纬度采样之后的数据，大小应等于Plot_x_range * Plot_y_range
Latitude_and_Longitude_Raw_Array = np.ones([Plot_x_range, Plot_y_range])
Plot_Latitude_List = []
Plot_Longitude_List = []
Plot_Available_List = []

for i in range(Latitude_and_Longitude_Raw_Array.shape[0]):
    for j in range(Latitude_and_Longitude_Raw_Array.shape[1]):
        Plot_Latitude_List.append(i * Plot_scale / Latitude_scale + Latitude_min)
        Plot_Longitude_List.append(- j * Plot_scale / Longitude_scale + Longitude_min)
        Plot_Available_List.append(1)
Data_Dict = {
    "Latitude": Plot_Latitude_List,
    "Longitude": Plot_Longitude_List,
    "Available": Plot_Available_List
}
Data_to_Save = DataFrame(Data_Dict)
# print(Latitude_and_Longitude_Array)
Data_to_Save.to_excel('program_output_sampling.xlsx')
