# VMD_2D_python
# Python3 implementation of 2D Variational Mode Decomposition using NumPy

Written by: Dodge(Lang HE) asdsay@gmail.com <br />
Updated date: 2023-11-16 <br />
Variational Mode Decomposition for Python in 2D

**VMD**, aka Variational Mode Decomposition, is a signal processing tool that decompse the input signal into different band-limited IMFs. <br />
**VMD_2D**, means we are processing 2D signal (Two dimension should usually have same length). Project **VMD_2D_Python**  is an imitation of [that in MATLAB](https://uk.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition?s_tid=mwa_osa_a). Spectrum-based decomposition of a 2D input signal into k band-separated modes. 

-------

In this project, I used a grey picture to test. The function ***VMD2D*** only needs Numpy, but we also need OpenCV and matplotlib to read and print the picture. 

![TestResule](Test_N100.png)


Please also pay attention to line 34-35 in *VMD2D.py* : <br />
`     # Maximum number of iterations`<br />
`     N = 3000`
<br /> In the original Matlab code, it was a solid **3000**. However under my test, the sample pictgure does **not** converge. Luckily, for this picture, it is practically the same if I set **N = 100**. So feel free to change the **N** value.

-------

If you are looking for document to describe Variational mode decomposition, please turn to the original paper:

K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Trans. on Signal Processing (in press)
please check here for update reference: 
http://dx.doi.org/10.1109/TSP.2013.2288675
