# YOLO V5
物体检测，包括手部检测。  

## 项目介绍    
### 手部检测  
手部检测示例如下 ：       
* 视频示例：  
![video](github.com/EricLee2021-72324/yolo-v5/raw/main/samples/handd.gif)    
## 项目配置  
* 作者开发环境：  
* Python 3.7  
* PyTorch >= 1.5.1  

## 数据集   
### 手部检测数据集   
该项目数据集采用 TV-Hand 和 COCO-Hand (COCO-Hand-Big 部分) 进行制作。  
TV-Hand 和 COCO-Hand数据集官网地址 http://vision.cs.stonybrook.edu/~supreeth/   
```   
感谢数据集贡献者。    
Paper：  
Contextual Attention for Hand Detection in the Wild. S. Narasimhaswamy, Z. Wei, Y. Wang, J. Zhang, and M. Hoai, IEEE International Conference on Computer Vision, ICCV 2019.   
```
* [该项目制作的训练集的数据集下载地址(百度网盘 Password: 25y3 )](https://pan.baidu.com/s/1y2A3sgNOS0V475QAXXgbTA)   

### 所有数据集的数据格式  
size是全图分辨率， (x，y) 是目标物体中心对于全图的归一化坐标，w,h是目标物体边界框对于全图的归一化宽、高。   

```  
dw = 1./(size[0])  
dh = 1./(size[1])  
x = (box[0] + box[1])/2.0 - 1  
y = (box[2] + box[3])/2.0 - 1  
w = box[1] - box[0]  
h = box[3] - box[2]  
x = x*dw  
w = w*dw  
y = y*dh  
h = h*dh  
```  

为了更好了解标注数据格式，可以通过运行 show_yolo_anno.py 脚本进行制作数据集的格式。注意配置脚本里的path和path_voc_names，path为标注数据集的相关文件路径，path_voc_names为数据集配置文件。
### 制作自己的训练数据集
* 如下所示,每一行代表一个物体实例，第一列是标签，后面是归一化的中心坐标(x,y),和归一化的宽(w)和高(h)，且每一列信息空格间隔。归一化公式如上，同时可以通过show_yolo_anno.py进行参数适配后，可视化验证其正确性。
```
label     x                  y                   w                  h
0 0.6200393316313977 0.5939000244140625 0.17241466452130497 0.14608001708984375
0 0.38552491996544863 0.5855700073242187 0.14937006832733554 0.1258599853515625
0 0.32889763138738515 0.701989990234375 0.031338589085055775 0.0671400146484375
0 0.760577424617577 0.69422998046875 0.028556443261975064 0.0548599853515625
0 0.5107086662232406 0.6921500244140625 0.018792660530470802 0.04682000732421875
0 0.9295538153861138 0.67602001953125 0.03884511231750328 0.01844000244140625
```

## 预训练模型   
### 从零开始预训练模型
* [预训练模型下载地址(百度网盘 Password: ad4l )](https://pan.baidu.com/s/1BuqU7XFRvRW8Rem4D_1U-w)  
### 手部检测预训练模型    
* 包括yolo_v5预训练模型图像输入尺寸640。  
* [预训练模型下载地址(百度网盘 Password: x7d4 )](https://pan.baidu.com/s/1b8-krpwlbw9cqYqtFUQGRQ)     

## 项目使用方法     

### 数据集可视化    
* 根目录下运行命令： show_yolo_anno.py   (注意脚本内相关参数配置 )   

### 模型训练     
* 根目录下运行命令： python train.py     (注意脚本内相关参数配置 )   

### 模型推理    
* 根目录下运行命令： python video.py   (注意脚本内相关参数配置  )  
