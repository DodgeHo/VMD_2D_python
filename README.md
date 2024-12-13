# VMD_2D_python
Python3 implementation of 2D Variational Mode Decomposition using NumPy

## Overview
Written by: Dodge(Lang HE) asdsay@gmail.com  
Updated date: 2024-12-13

**VMD** (Variational Mode Decomposition) is a signal processing tool that decomposes input signals into different band-limited IMFs (Intrinsic Mode Functions).

**VMD_2D** processes 2D signals (both dimensions should typically have the same length). This project **VMD_2D_Python** is an implementation based on [the MATLAB version](https://uk.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition?s_tid=mwa_osa_a), providing spectrum-based decomposition of a 2D input signal into k band-separated modes.

## Dependencies
- NumPy (required for core VMD2D function)
- OpenCV (optional, for image reading)
- Matplotlib (optional, for visualization)

## Usage Notes
In this project, a grayscale image is used for testing.

![TestResult](Test_N100.png)

### Important Configuration
In *VMD2D.py*, lines 34-35:

```python
# Maximum number of iterations
N = 3000
```
While the original MATLAB code used a fixed value of **3000**, our testing showed that the sample image does not converge at this value. However, setting **N = 100** provides practically equivalent results for this particular image. Feel free to adjust the **N** value based on your needs.

## References
For detailed information about Variational Mode Decomposition, please refer to the original paper:

K. Dragomiretskiy, D. Zosso, "Variational Mode Decomposition", IEEE Trans. on Signal Processing  
DOI: http://dx.doi.org/10.1109/TSP.2013.2288675

# 二维VMD（变分模态分解）的Python3实现，使用NumPy

## 概述
作者：Dodge asdsay@gmail.com 更新日期：2023-11-16

**VMD**（变分模态分解）是一种信号处理算法，可以将输入信号分解为不同带限的内禀模态函数（IMFs）。
**VMD_2D**意味着我们正在处理二维信号（通常两个维度应该长度相同）。项目是MATLAB中实现的模仿。基于频谱的二维输入信号分解为k个带分离模式。
本项目**VMD_2D_Python**是参考于[其在MATLAB中的实现](https://uk.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition?s_tid=mwa_osa_a)。基于频谱的二维输入信号分解为k个带分离模式。

## 依赖项
- NumPy（核心VMD2D函数所需）
- OpenCV（可选，用于图像读取）
- Matplotlib（可选，用于可视化）

## 使用说明
在这个项目中，我用一张灰度图片进行测试。本项目**VMD_2D_Python**仅需要Numpy，但我们还需要OpenCV和matplotlib两个库来读取和显示图片。

![测试结果](Test_N100.png)

### 重要配置
在 *VMD2D.py*第34-35行：

```python
# Maximum number of iterations
N = 3000
```
在原始的Matlab代码中，N是固定值3000。然而在测试中，样本图片计算的误差没有收敛。发现对于这张图片，如果设置N = 100，实际效果几乎就收敛了。请用户更改N值测试效果。

## 参考文献
如果需要描述变分模态分解的文档，可参阅原始论文：
K. Dragomiretskiy, D. Zosso, "Variational Mode Decomposition", IEEE Trans. on Signal Processing  
DOI: http://dx.doi.org/10.1109/TSP.2013.2288675
