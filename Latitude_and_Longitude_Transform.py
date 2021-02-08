# This is a latitude and longitude transform script
import pandas as pd
import numpy as np
import math

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
# 用于存放经纬度原始数据
Latitude_and_Longitude_Raw_Array = np.zeros([Latitude_and_Longitude_Data.shape[0], 2])
# 用于存放经纬度转化之后的数据以及对应的Positive判断值
Latitude_and_Longitude_Array = np.zeros([Latitude_and_Longitude_Data.shape[0], 3])
# 定义绘图坐标系的范围
Plot_scale = 2
Plot_x_range = abs(round((Latitude_max - Latitude_min) * Latitude_scale / Plot_scale))
Plot_y_range = abs(round((Longitude_max - Longitude_min) * Longitude_scale / Plot_scale))
print(Plot_x_range, Plot_y_range)

Population_Number = (Latitude_and_Longitude_Data['Lab Status']).tolist().count('Positive ID')
Population_Point = np.zeros([Population_Number, 2])
Point_count = 0
# 经纬度信息转换到绘图坐标系上面
for i in range(Latitude_and_Longitude_Data.shape[0]):
    Latitude_and_Longitude_Raw_Array[i][0] = Latitude_and_Longitude_Data.loc[i, ['Latitude']]
    Latitude_and_Longitude_Raw_Array[i][1] = Latitude_and_Longitude_Data.loc[i, ['Longitude']]
    Latitude_and_Longitude_Array[i][0] = abs(round(float(Latitude_and_Longitude_Data.loc[i, ['Latitude']] - Latitude_min) * Latitude_scale / Plot_scale))
    Latitude_and_Longitude_Array[i][1] = abs(round(float(Latitude_and_Longitude_Data.loc[i, ['Longitude']] - Longitude_min) * Longitude_scale / Plot_scale))
    if (Latitude_and_Longitude_Data.loc[i, ['Lab Status']]).tolist() == ['Positive ID']:
        Latitude_and_Longitude_Array[i][2] = 1
        Population_Point[Point_count][0] = Latitude_and_Longitude_Array[i][0]
        Population_Point[Point_count][1] = Latitude_and_Longitude_Array[i][1]
        Point_count += 1
    else:
        Latitude_and_Longitude_Array[i][2] = 0
# print(Latitude_and_Longitude_Array)

print(Population_Point)
