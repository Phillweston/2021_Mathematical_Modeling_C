# This is a image classification script
import pandas as pd
import os
import shutil

Image_by_GlobalID = pd.read_excel('2021MCM_ProblemC_ Images_by_GlobalID.xlsx')
Dataset = pd.read_excel('2021MCMProblemC_DataSet.xlsx')
# print(Image_by_GlobalID.head(5))
index = Dataset.index[Dataset['Lab Status'] == 'Positive ID']
GlobalID_Selected = []
ImageName_Selected = []
ImageName_Str_temp = []

for i in range(len(index)):
    # 注意这里存在多个GlobalID对应一个文件的情况
    GlobalID_Str = (Dataset.loc[index[i], ['GlobalID']].to_dict())['GlobalID']
    GlobalID_Selected.append(GlobalID_Str)
    index_image = Image_by_GlobalID.index[Image_by_GlobalID['GlobalID'] == GlobalID_Str]
    ImageName_Str = list((Image_by_GlobalID.loc[index_image, ['FileName']].to_dict())['FileName'].values())
    for j in range(len(ImageName_Str)):
        ImageName_Selected = ImageName_Selected + ImageName_Str[j].split(",")
    # ImageName_Selected.append(ImageName_Str)

print(GlobalID_Selected)
print(ImageName_Selected)

image_source_dir_path = os.getcwd() + '\\2021MCM_ProblemC_Files_JPG'
image_destination_dir_path = os.getcwd() + '\\Image_Positive'
print(image_source_dir_path)
print(image_destination_dir_path)

for i in range(len(ImageName_Selected)):
    image_name = os.path.splitext(ImageName_Selected[i])[0] + '.jpg'
    print(image_name)
    image_source_path = os.path.join(image_source_dir_path, image_name)
    image_destination_path = os.path.join(image_destination_dir_path, image_name)
    shutil.copy(image_source_path, image_destination_path)
