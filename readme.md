# 使用方法
## 准备环境
- Anaconda 3.7
- Pytorch 1.4+（及其对应的torchvision）
- scikit-learn（sklearn）
- numpy
- tensorboard
- SimpleITK
- python-opencv


## 准备数据集
- 在`根`目录中创建`data`文件夹，并将数据集存放在`data`文件夹中，并确保目录结构为：
```python
├─data
│  ├─CT       # 存放CT图像
│  │  ├─volume-0.nii
│  │  ├─volume-1.nii
│  │  ├─volume-2.nii
│  │  └─...
│  ├─seg       # 存放mask图像
│     ├─segmentation-0.nii
│     ├─segmentation-1.nii
│     ├─segmentation-2.nii
│     └─...
```
## 预处理
- 使用以下命令，或是使用vscode运行：`preprocessing.py`
```shell
python preprocessing.py
```
## 训练
- 生成的日志位于`logs`文件夹下  
- 生成的模型位于`checkpoint`文件夹下
- 使用以下命令，或是使用vscode的调试配置：`训练`  
```shell
python train.py --learning_rate=0.001 --workers=0 --epochs=80 --batch_size=2 --num_steps_to_display=500 -d=data
```
## 评估
- 使用以下命令，或是使用vscode的调试配置：`评估`  
```
python eval.py --workers=4 --model_path=checkpoint/checkpoint-epoch0080.pth -d=data
```
## 推断
- 生成的推断结果位于`outputs`目录下
- 使用以下命令，或是使用vscode的调试配置：`推断（使用COCO聚类标签）`  
```
python infer.py --model_path=checkpoint/checkpoint-epoch0080.pth --input_dir=data/final_CT --output_dir=outputs
```

